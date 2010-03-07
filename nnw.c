/*  NeuralNetworker - Version 0.1 
    Copyright 2010 Daniele Iamartino

    This file is part of NeuralNetWorker.

    NeuralNetWorker is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#include "nnw.h"

#define VERSION         "0.1"



typedef enum {False, True} boolean;
int verbose = 0;

struct {
    int inputNeurons;
    int hiddenNeurons;
    int outputNeurons;
    int hiddenLayers;
    
    boolean useInputBias;
    boolean useHiddenBias;
    boolean useMomentum;
    
    int inputBias;
    int hiddenBias;
    
    networkPrecision inputBiasValue;
    networkPrecision hiddenBiasValue;
    
    networkPrecision *inputActivation;
    networkPrecision **hiddenActivation;
    networkPrecision *outputActivation;
    
    networkPrecision **inputWeights;
    networkPrecision ***hiddenWeights;
    networkPrecision **outputWeights;
    
    networkPrecision **inputMomentum;
    networkPrecision ***hiddenMomentum;
    networkPrecision **outputMomentum;
    
    networkPrecision **hiddenDeltas;
    networkPrecision *outputDeltas;
    
    networkPrecision (*function)(networkPrecision); 
    networkPrecision (*derivedFunction)(networkPrecision);
    networkPrecision (*inputWeightsValue)();
    networkPrecision (*hiddenWeightsValue)();
    networkPrecision (*outputWeightsValue)();
} neuralNetwork;    

struct {
    int quantity; /* This specify how much examples there are */
    networkPrecision **inputs;
    networkPrecision **outputs;
} networkExamples;

networkPrecision randomGenerator(double min, double max)
{
    double range = max - min;
    double rnd = (networkPrecision)rand() / RAND_MAX;
  
    return min + rnd * range;
}

networkPrecision randomizer()
{
    return randomGenerator(-0.2,0.2);
}
networkPrecision randomizer2()
{
    return randomGenerator(-2.0,2.0);
}

networkPrecision transferFunction_Sigmodial(networkPrecision input)
{
    return ((1.0)/(1.0 + pow(M_E, -input)));
}

networkPrecision transferFunction_SigmodialDerived(networkPrecision input)
{
    return ( input - pow(input, 2) );
}

networkPrecision transferFunction_Step(networkPrecision input)
{
    if (input<=0.0)
        return 0.0;
    else
        return 1.0;
}

networkPrecision transferFunction_StepDerived(networkPrecision input)
{
    return 0.0;
}

networkPrecision transferFunction_Tanh(networkPrecision input)
{
    return tanh(input);
}

networkPrecision transferFunction_TanhDerived(networkPrecision input)
{
    return (1.0 - pow(input,2));
}


void weightsDeallocatorInput(int start)
{
    int i;
    
    if ( start == -1 )
        start = neuralNetwork.inputNeurons + neuralNetwork.inputBias -1;
    
    for (i=start;i>=0;i--)
        free(neuralNetwork.inputWeights[i]);
    
    free(neuralNetwork.inputWeights);
}

void weightsDeallocatorInputMomentum(int start)
{
    int i;
    
    if ( start == -1 )
        start = neuralNetwork.inputNeurons + neuralNetwork.inputBias -1;
    
    for (i=start;i>=0;i--)
        free(neuralNetwork.inputMomentum[i]);
    
    free(neuralNetwork.inputMomentum);
    weightsDeallocatorInput(-1);
}

void weightsDeallocatorHidden(int start1, int start2)
{
    int i,j;
    
    if ( start1 == -1 )
        start1 = neuralNetwork.hiddenLayers - 2;
    
    if ( start2 == -1 )
        start2 = neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias - 1;
    
    for (i=start1;i>=0;i--)
    {
        for (j=start2;j>=0;j--)
        {
            free(neuralNetwork.hiddenWeights[i][j]);
        }
        free(neuralNetwork.hiddenWeights[i]);
    }
    free(neuralNetwork.hiddenWeights);
    weightsDeallocatorInputMomentum(-1);
}

void weightsDeallocatorHiddenMomentum(int start1, int start2)
{
    int i,j;
    
    if ( start1 == -1 )
        start1 = neuralNetwork.hiddenLayers - 2;
    
    if ( start2 == -1 )
        start2 = neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias - 1;
    
    for (i=start1;i>=0;i--)
    {
        for (j=start2;j>=0;j--)
        {
            free(neuralNetwork.hiddenMomentum[i][j]);
        }
        free(neuralNetwork.hiddenMomentum[i]);
    }
    free(neuralNetwork.hiddenMomentum);
    
    weightsDeallocatorHidden(-1,-1);
}

void weightsDeallocatorOutput(int start)
{
    int i;
    
    if ( start == -1 )
        start = neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias -1;
    for (i=start;i>=0;i--)
        free(neuralNetwork.outputWeights[i]);
    free(neuralNetwork.outputWeights);
    
    weightsDeallocatorHiddenMomentum(-1,-1);
}

void weightsDeallocatorOutputMomentum(int start)
{
    int i;
    
    if ( start == -1 )
        start = neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias -1;
    for (i=start;i>=0;i--)
        free(neuralNetwork.outputMomentum[i]);
    free(neuralNetwork.outputMomentum);
    
    weightsDeallocatorOutput(-1);
}

void weightsDeallocator()
{
    weightsDeallocatorOutputMomentum(-1);
}

int weightsAllocator()
{
    /* It reads network parameters from the structure and it allocate
     * the necessary space in the HEAP area for weights and momentum values.
     */

    
    int i,j;
    
    /* Input-Hidden area (Weights) */
    neuralNetwork.inputWeights = malloc(sizeof(networkPrecision *)*(neuralNetwork.inputNeurons + neuralNetwork.inputBias));
    if (neuralNetwork.inputWeights == NULL)
        return 1;
    for (i=0;i<(neuralNetwork.inputNeurons + neuralNetwork.inputBias);i++)
    {
        neuralNetwork.inputWeights[i] = malloc(sizeof(networkPrecision)*neuralNetwork.hiddenNeurons);
        if (neuralNetwork.inputWeights[i] == NULL)
        {
            weightsDeallocatorInput(i-1);
            return 1;
        }
    }
    /* Input-Hidden area (Momentum) */
    neuralNetwork.inputMomentum = malloc(sizeof(networkPrecision *)*(neuralNetwork.inputNeurons + neuralNetwork.inputBias));
    if (neuralNetwork.inputMomentum == NULL)
    {
        weightsDeallocatorInput(-1);
        return 1;
    }
    for (i=0;i<(neuralNetwork.inputNeurons + neuralNetwork.inputBias);i++)
    {
        neuralNetwork.inputMomentum[i] = malloc(sizeof(networkPrecision)*neuralNetwork.hiddenNeurons);
        if (neuralNetwork.inputMomentum[i] == NULL)
        {
            weightsDeallocatorInputMomentum(i-1);
            return 1;
        }
    }
    
    
    if (neuralNetwork.hiddenLayers>1)
    {
        /* Hidden-Hidden area (Weights) */
        neuralNetwork.hiddenWeights = malloc(sizeof(networkPrecision **)*(neuralNetwork.hiddenLayers - 1));
        if (neuralNetwork.hiddenWeights == NULL)
        {
            weightsDeallocatorInputMomentum(-1);
            return 1;
        }
        for (i=0;i<(neuralNetwork.hiddenLayers - 1);i++)
        {
            neuralNetwork.hiddenWeights[i] = malloc(sizeof(networkPrecision *)*(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias));
            if (neuralNetwork.hiddenWeights[i] == NULL)
            {
                weightsDeallocatorHidden(i-1,-1);
                return 1;
            }
            for (j=0;j<(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias);j++)
            {
                neuralNetwork.hiddenWeights[i][j] = malloc(sizeof(networkPrecision)*(neuralNetwork.hiddenNeurons));
                if (neuralNetwork.hiddenWeights[i][j] == NULL)
                {
                    weightsDeallocatorHidden(i-1,j-1);
                    return 1;
                }
            }
        }
        
        /* Hidden-Hidden area (Momentum) */
        neuralNetwork.hiddenMomentum = malloc(sizeof(networkPrecision **)*(neuralNetwork.hiddenLayers - 1));
        if (neuralNetwork.hiddenMomentum == NULL)
        {
            weightsDeallocatorHidden(-1,-1);
            return 1;
        }
        for (i=0;i<(neuralNetwork.hiddenLayers - 1);i++)
        {
            neuralNetwork.hiddenMomentum[i] = malloc(sizeof(networkPrecision *)*(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias));
            if (neuralNetwork.hiddenMomentum[i] == NULL)
            {
                weightsDeallocatorHiddenMomentum(i-1,-1);
                return 1;
            }
            for (j=0;j<(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias);j++)
            {
                neuralNetwork.hiddenMomentum[i][j] = malloc(sizeof(networkPrecision)*(neuralNetwork.hiddenNeurons));
                if (neuralNetwork.hiddenMomentum[i][j] == NULL)
                {
                    weightsDeallocatorHiddenMomentum(i-1,j-1);
                    return 1;
                }
            }
        }

    }
    
    /* Hidden-Output area (Weights) */
    neuralNetwork.outputWeights = malloc(sizeof(networkPrecision *)*(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias));
    if (neuralNetwork.outputWeights == NULL)
    {
        weightsDeallocatorHiddenMomentum(-1,-1);
        return 1;
    }
    for (i=0;i<(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias);i++)
    {
        neuralNetwork.outputWeights[i] = malloc(sizeof(networkPrecision)*neuralNetwork.outputNeurons);
        if (neuralNetwork.outputWeights[i] == NULL)
        {
            weightsDeallocatorOutput(i-1);
            return 1;
        }
    }
    /* Hidden-Output area (Momentum) */
    neuralNetwork.outputMomentum = malloc(sizeof(networkPrecision *)*(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias));
    if (neuralNetwork.outputMomentum == NULL)
    {
        weightsDeallocatorOutput(-1);
        return 1;
    }
    for (i=0;i<(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias);i++)
    {
        neuralNetwork.outputMomentum[i] = malloc(sizeof(networkPrecision)*neuralNetwork.outputNeurons);
        if (neuralNetwork.outputMomentum[i] == NULL)
        {
            weightsDeallocatorOutputMomentum(i-1);
            return 1;
        }
    }
    
    return 0;
}

void activationDeallocatorInput()
{
    free(neuralNetwork.inputActivation);
}

void activationDeallocatorHidden(int start) 
{
    int i;
    if ( start == -1 )
        start = neuralNetwork.hiddenLayers-1;
    
    for (i=start; i>=0; i--)
        free(neuralNetwork.hiddenActivation[i]);
    
    free(neuralNetwork.hiddenActivation);
    activationDeallocatorInput();
}

void activationDeallocatorOutput()
{
    free(neuralNetwork.outputActivation);
    activationDeallocatorHidden(-1);
}

void activationDeallocator()
{
    activationDeallocatorOutput();
}

int activationAllocator()
{
    /* It reads network parameters from the structure and it allocate
     * the necessary space in the HEAP area for activation values.
     */
    
    int i;
    
    /* Setting up Activation space */
    /* Input layer */
    
    neuralNetwork.inputActivation = malloc(sizeof(networkPrecision)*(neuralNetwork.inputNeurons + neuralNetwork.inputBias));
    if ( neuralNetwork.inputActivation == NULL )
        return 1;
    
    /* Hidden layer */
    neuralNetwork.hiddenActivation = malloc(sizeof(networkPrecision *)*neuralNetwork.hiddenLayers);
    if ( neuralNetwork.hiddenActivation == NULL )
    {
        activationDeallocatorInput();
        return 1;
    }
    
    for (i=0; i<neuralNetwork.hiddenLayers; i++)
    {
        neuralNetwork.hiddenActivation[i] = malloc(sizeof(networkPrecision)*(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias));
        if ( neuralNetwork.hiddenActivation[i] == NULL )
        {
            activationDeallocatorHidden(i-1);
            return 1;
        }
    } 
    
    /* Output layer */
    neuralNetwork.outputActivation = malloc(sizeof(networkPrecision)*(neuralNetwork.outputNeurons));
    if ( neuralNetwork.outputActivation == NULL )
    {
        activationDeallocatorHidden(-1);
        return 1;
    }
    
    return 0;
}

void setBias()
{
    /* This sets to 1 all bias neurons*/
    int i;
    
    if ( neuralNetwork.useInputBias == True )
        neuralNetwork.inputActivation[neuralNetwork.inputNeurons] = neuralNetwork.inputBiasValue;
    
    if ( neuralNetwork.useHiddenBias == True )
        for (i=0;i<neuralNetwork.hiddenLayers;i++)
            neuralNetwork.hiddenActivation[i][neuralNetwork.hiddenNeurons] = neuralNetwork.hiddenBiasValue;
    
}

void flushWeights()
{

    int i,j,k;
    
    /* Input-Hidden */
    for (i=0;i<(neuralNetwork.inputNeurons+neuralNetwork.inputBias);i++)
    {
        for (j=0;j<(neuralNetwork.hiddenNeurons);j++)
        {
            neuralNetwork.inputWeights[i][j] = neuralNetwork.inputWeightsValue();
            neuralNetwork.inputMomentum[i][j] = 0;
        }
    }
    
    /* Hidden-Hidden */
    for (i=0;i<(neuralNetwork.hiddenLayers-1);i++)
    {
        for (j=0;j<(neuralNetwork.hiddenNeurons+neuralNetwork.hiddenBias);j++)
        {
            for (k=0;k<(neuralNetwork.hiddenNeurons);k++)
            {
                neuralNetwork.hiddenWeights[i][j][k] = neuralNetwork.hiddenWeightsValue();
                neuralNetwork.hiddenMomentum[i][j][k] = 0;
            }
        }
    }
        
    /* Hidden-Output */
    for (i=0;i<(neuralNetwork.hiddenNeurons+neuralNetwork.hiddenBias);i++)
    {
        for (j=0;j<(neuralNetwork.outputNeurons);j++)
        {
            neuralNetwork.outputWeights[i][j] = neuralNetwork.outputWeightsValue();
            neuralNetwork.outputMomentum[i][j] = 0;
        }
    }
}

void updateNetwork()
{
    /*  This function is called after changing activation values
        of the INPUT layer */
    int i,j,k;
    networkPrecision sum;
    
    /* Calculate new activation values for the first hidden layer */
    for (i=0;i<(neuralNetwork.hiddenNeurons);i++)
    {
        sum = 0.0;
        for (j=0;j<(neuralNetwork.inputNeurons+neuralNetwork.inputBias);j++)
            sum = sum + (neuralNetwork.inputWeights[j][i]*neuralNetwork.inputActivation[j]);
        neuralNetwork.hiddenActivation[0][i] = neuralNetwork.function(sum); /* Correct? */
    }
    
    if (neuralNetwork.hiddenLayers>1)
    {
        /* Calculate new activation values for the optionals hidden layers */
        for (i=0;i<(neuralNetwork.hiddenLayers-1);i++) 
        {
            for (j=0;j<(neuralNetwork.hiddenNeurons);j++)
            {
                sum = 0.0;
                for (k=0;k<(neuralNetwork.hiddenNeurons+neuralNetwork.hiddenBias);k++)
                    sum = sum + (neuralNetwork.hiddenWeights[i][k][j]*neuralNetwork.hiddenActivation[i][k]);
                neuralNetwork.hiddenActivation[i][j] = neuralNetwork.function(sum);
            }
        }
    }
    
    /* Calculate new activation values for the output layer */
    for (i=0;i<(neuralNetwork.outputNeurons);i++)
    {
        sum = 0.0;
        for (j=0;j<(neuralNetwork.hiddenNeurons+neuralNetwork.hiddenBias);j++)
            sum = sum + (neuralNetwork.outputWeights[j][i]*
                neuralNetwork.hiddenActivation[neuralNetwork.hiddenLayers-1][j]);
        neuralNetwork.outputActivation[i] = neuralNetwork.function(sum);
    }
}

void deltaDeallocatorHidden(int start)
{
    int i;
    
    if ( start == -1 )
        start = neuralNetwork.hiddenLayers - 1;
    
    for (i=start;i>=0;i--)
        free(neuralNetwork.hiddenDeltas[i]);
    free(neuralNetwork.hiddenDeltas);
}

void deltaDeallocatorOutput()
{
    free(neuralNetwork.outputDeltas);
    deltaDeallocatorHidden(-1);
}

void deltaDeallocator()
{
    deltaDeallocatorOutput();
}

int deltaAllocator()
{
    int i;
    
    /* Setting up delta space */
    /* Just Hiddens and Output, not input layer */
    
    /* Hidden layer */
    neuralNetwork.hiddenDeltas = malloc(sizeof(networkPrecision *)*neuralNetwork.hiddenLayers);
    if ( neuralNetwork.hiddenDeltas == NULL )
        return 1;
    
    for (i=0; i<neuralNetwork.hiddenLayers; i++)
    {
        neuralNetwork.hiddenDeltas[i] = malloc(sizeof(networkPrecision)*(neuralNetwork.hiddenNeurons));
        if ( neuralNetwork.hiddenDeltas[i] == NULL )
        {
            deltaDeallocatorHidden(i-1);
            return 1;
        }
    } 
    
    /* Output layer */
    neuralNetwork.outputDeltas = malloc(sizeof(networkPrecision)*(neuralNetwork.outputNeurons));
    if ( neuralNetwork.outputDeltas == NULL )
    {
        deltaDeallocatorHidden(-1);
        return 1;
    }
    
    return 0;
}

errorPrecision backPropagation(float learningRate, float momentumFactor, int i)
{
    /* Calculate deltas and change weights */
    int j,k,h;
    networkPrecision error;
    networkPrecision change;
    errorPrecision totalerror;
    
    /* Output deltas */
    for (j=0;j<neuralNetwork.outputNeurons;j++)
    {
        error = ( networkExamples.outputs[i][j] - neuralNetwork.outputActivation[j] );
        neuralNetwork.outputDeltas[j] = ( neuralNetwork.derivedFunction(neuralNetwork.outputActivation[j]) * error );
    }
    
    /* Hidden deltas */
    /* First just last hidden layer */
    for (j=0;j<neuralNetwork.hiddenNeurons;j++)
    {
        error = 0.0;
        for (k=0;k<neuralNetwork.outputNeurons;k++)
        {
            error = error + neuralNetwork.outputDeltas[k]*neuralNetwork.outputWeights[j][k];
        }
        neuralNetwork.hiddenDeltas[neuralNetwork.hiddenLayers-1][j] =
            ( neuralNetwork.derivedFunction(neuralNetwork.hiddenActivation[neuralNetwork.hiddenLayers-1][j]) * error );
    }
    
    /* If the net does have more than 1 hidden layer */
    if (neuralNetwork.hiddenLayers>1)
    {
        for (h=(neuralNetwork.hiddenLayers-2);h>=0;h--)
        {
            for (j=0;j<neuralNetwork.hiddenNeurons;j++)
            {
                error = 0.0;
                for (k=0;k<neuralNetwork.hiddenNeurons;k++)
                {
                    error = error + neuralNetwork.hiddenDeltas[h+1][k]*neuralNetwork.hiddenWeights[h][j][k];
                }
                neuralNetwork.hiddenDeltas[h][j] = 
                    ( neuralNetwork.derivedFunction(neuralNetwork.hiddenActivation[h][j]) * error );
            }
        }
    }
    
    /* Change ALL network weights */
    /* Change network weights Hidden-Output */
    for (j=0;j<(neuralNetwork.hiddenNeurons+neuralNetwork.hiddenBias);j++)
    {
        for (k=0;k<(neuralNetwork.outputNeurons);k++)
        {
            change = neuralNetwork.outputDeltas[k]*neuralNetwork.hiddenActivation[neuralNetwork.hiddenLayers-1][j];
            neuralNetwork.outputWeights[j][k] = neuralNetwork.outputWeights[j][k] +
                ( learningRate*change ) + ( momentumFactor*neuralNetwork.outputMomentum[j][k] );
            neuralNetwork.outputMomentum[j][k] = change;
        }
    }
    
    /* Change network weights Hidden-Hidden */
    if (neuralNetwork.hiddenLayers>1)
    {
        for (h=(neuralNetwork.hiddenLayers-2);h>=0;h--)
        {
            for (j=0;j<(neuralNetwork.hiddenNeurons+neuralNetwork.hiddenBias);j++)
            {
                for (k=0;k<neuralNetwork.hiddenNeurons;k++)
                {
                    change = neuralNetwork.hiddenDeltas[h+1][k]*neuralNetwork.hiddenActivation[h][j];
                    neuralNetwork.hiddenWeights[h][j][k] = neuralNetwork.hiddenWeights[h][j][k] 
                        + ( learningRate*change ) + ( momentumFactor*neuralNetwork.hiddenMomentum[h][j][k] );
                    neuralNetwork.hiddenMomentum[h][j][k] = change;
                }
            }
        }
    }
    
    /* Change network weights Input-Hidden */
    for (k=0;k<(neuralNetwork.inputNeurons+neuralNetwork.inputBias);k++)
    {
        for (j=0;j<(neuralNetwork.hiddenNeurons);j++)
        {
            change = neuralNetwork.hiddenDeltas[0][j]*neuralNetwork.inputActivation[k];
            neuralNetwork.inputWeights[k][j] = neuralNetwork.inputWeights[k][j] 
                + ( learningRate*change ) + ( momentumFactor*neuralNetwork.inputMomentum[k][j] );
            neuralNetwork.inputMomentum[k][j] = change;
        }
    }
    
    /* Calculate medium quadratic error */
    totalerror = 0.0;
    for (k=0;k<neuralNetwork.outputNeurons;k++)
        totalerror = totalerror + 0.5*
            ((errorPrecision) pow((networkExamples.outputs[i][k]-neuralNetwork.outputActivation[k]), 2));
    
    return totalerror;
    
    
}

errorPrecision trainNetwork(float learningRate, float momentumFactor, int iterations, errorPrecision minError)
{
    int i,j;
    long int k;

    errorPrecision error;
    k=0;
    do
    {
        k++;
        error = 0.0;
        for (i=0;i<networkExamples.quantity;i++)
        {
            /* Insert the current example in the network and then call the update function */
            /* I'm supposing that examples are in the correct number */
            for (j=0;j<(neuralNetwork.inputNeurons);j++)
                neuralNetwork.inputActivation[j] = networkExamples.inputs[i][j];
            updateNetwork();
            error = error + backPropagation(learningRate, momentumFactor, i);
            if (verbose>1)
                printf("    [*] Iteration %ld , error sum = %.10f\n", k, error);
        }
        if (verbose>0)
            {
                printf("Iteration %ld, error sum = %.10f\n", k, error);
                if (verbose>1)
                    printf("\n");
            }
    }
    while (error>minError && k!=iterations);
    
    return error;
    
}


int main(int argc, char *argv[])
{
    /* Here the program recive arguments from command line
     * and it set the network structure. It also checks if
     * network parameters are right.
     */
    int b,c;
    int iterations=-1;
    errorPrecision error, required=0.0;
    time_t rawtime1, rawtime2;
    int option_index = 0;
    static const struct option long_options[] = {
	{ "help", 	no_argument,	        NULL, 'h' },
        { "verbose",	no_argument,	        NULL, 'v' },
        { "error",	required_argument,	NULL, 'e' },
        { "iterations",	required_argument,	NULL, 'i' },
	{ 0, 0, 0, 0 }
    };
    
    srand(0);
    do {
        c = getopt_long(argc, argv, "hve:i:", long_options, &option_index);
        switch (c)
        {
            case 'h':
                printf("NeuralNetWorker - Version %s\nUsage: nnw [-options]\n\n", VERSION);
                printf("Options:\n");
                printf("\t--help or -h\t\t\t\t: Shows this help\n");
                printf("\t--verbose or -v\t\t\t\t: Verbose mode\n");
                printf("\t--error [value] or -e [value]\t\t: Sets the error required\n");
                printf("\t--iterations [value] or -i [value]\t: Sets the number of iterations to be done\n");
                exit(0);
                break;
            case 'v':
                verbose++;
                break;
            case 'e':
                required = atof(optarg);
                if (required<=0.0)
                {
                    printf("Invalid value for the --error parameter\n");
                    exit(1);
                }
                break;
            case 'i':
                iterations=atoi(optarg);
                if (iterations<=0)
                {
                    printf("Invalid value for the --iterations parameter\n");
                    exit(1);
                }
                break;
        }
    }
    while (c!=-1);
    if (argc==1)
    {
        printf("NeuralNetWorker - Version %s\nUsage: nnw [-options]\n\n", VERSION);
        exit(1);
    }
    neuralNetwork.inputNeurons=2;
    neuralNetwork.hiddenNeurons=2;
    neuralNetwork.outputNeurons=1;
    neuralNetwork.hiddenLayers=1;
    
    neuralNetwork.useHiddenBias=True;
    neuralNetwork.useInputBias=True;
    neuralNetwork.useMomentum=True; 
    
    neuralNetwork.inputBiasValue = 1.0;
    neuralNetwork.hiddenBiasValue = 1.0;

    neuralNetwork.function = transferFunction_Tanh;
    neuralNetwork.derivedFunction = transferFunction_TanhDerived;
    
    neuralNetwork.inputWeightsValue = randomizer;
    neuralNetwork.hiddenWeightsValue = randomizer2;
    neuralNetwork.outputWeightsValue = randomizer2;
    
    /* Bias neurons presence */
    if (neuralNetwork.useInputBias == True)
        neuralNetwork.inputBias = 1;
    else
        neuralNetwork.inputBias = 0;
    
    if (neuralNetwork.useHiddenBias == True)
        neuralNetwork.hiddenBias = 1;
    else
        neuralNetwork.hiddenBias = 0;
    
    if (activationAllocator() != 0)
        exit(1);
    if (weightsAllocator() != 0)
    {
        activationDeallocator();
        exit(1);
    }
    if (deltaAllocator() != 0)
    {
        activationDeallocator();
        weightsDeallocator();
        exit(1);
    }
    
    setBias();
    flushWeights();
    
    networkExamples.quantity = 4;
    networkExamples.inputs = malloc(sizeof(double *)*4);
    networkExamples.outputs = malloc(sizeof(double *)*4);
    for (b=0;b<4;b++)
    {
        networkExamples.inputs[b] = malloc(sizeof(double)*2);
        networkExamples.outputs[b] = malloc(sizeof(double));
    }
    networkExamples.inputs[0][0] = 0.0;
    networkExamples.inputs[0][1] = 0.0;
    networkExamples.outputs[0][0] = 0.0;
    
    networkExamples.inputs[1][0] = 0.0;
    networkExamples.inputs[1][1] = 1.0;
    networkExamples.outputs[1][0] = 1.0;
    
    networkExamples.inputs[2][0] = 1.0;
    networkExamples.inputs[2][1] = 0.0;
    networkExamples.outputs[2][0] = 1.0;
    
    networkExamples.inputs[3][0] = 1.0;
    networkExamples.inputs[3][1] = 1.0;
    networkExamples.outputs[3][0] = 0.0;
    
    time(&rawtime1);
    error = trainNetwork(0.5, 0.1, iterations, required);
    time(&rawtime2);
    
    printf("Training completed.\n Network error: %e\n Time elapsed: %d seconds\n", error, (int) (rawtime2-rawtime1));
    
    neuralNetwork.inputActivation[0] = 1.0;
    neuralNetwork.inputActivation[1] = 0.0;
    updateNetwork();
    printf("Output with 1 and 0 (waiting 1) -> %.16f\n",neuralNetwork.outputActivation[0]);
    
    neuralNetwork.inputActivation[0] = 0.0;
    neuralNetwork.inputActivation[1] = 1.0;
    updateNetwork();
    printf("Output with 0 and 1 (waiting 1) -> %.16f\n",neuralNetwork.outputActivation[0]);
    
    neuralNetwork.inputActivation[0] = 0.0;
    neuralNetwork.inputActivation[1] = 0.0;
    updateNetwork();
    printf("Output with 0 and 0 (waiting 0) -> %.16f\n",neuralNetwork.outputActivation[0]);
    
    neuralNetwork.inputActivation[0] = 1.0;
    neuralNetwork.inputActivation[1] = 1.0;
    updateNetwork();
    printf("Output with 1 and 1 (waiting 0) -> %.16f\n",neuralNetwork.outputActivation[0]);
    
    /* Deallocation part */
    activationDeallocator();
    weightsDeallocator();
    deltaDeallocator();
    
    exit(1);
}
