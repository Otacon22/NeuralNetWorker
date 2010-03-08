/*  NeuralNetworker - Version 0.1 
    Copyright 2010 Daniele Iamartino

    This file is part of NeuralNetWorker.

    NeuralNetWorker is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    NeuralNetWorker is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with NeuralNetWorker.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdio.h>
#include "nnw.h"

int netFileHeaderParser(FILE *fd)
{
    int i;
    
    if (fscanf(fd,"%d %d %d %d\n%d\n%d\n",&(neuralNetwork.inputNeurons),
    &(neuralNetwork.hiddenNeurons), &(neuralNetwork.outputNeurons),
    &(neuralNetwork.hiddenLayers), &(neuralNetwork.inputBias), 
    &(neuralNetwork.hiddenBias)) < 0)
        return 2;
    
    if (fscanf(fd,"%d\n\n",&i) < 0)
        return 3;
    
    if (neuralNetwork.inputBias == 0)
        neuralNetwork.useInputBias = False;
    
    else if (neuralNetwork.inputBias == 1)
        neuralNetwork.useInputBias = True;
    
    else
        return 4;
    
    if (neuralNetwork.hiddenBias == 0)
        neuralNetwork.useHiddenBias = False;
    
    else if (neuralNetwork.hiddenBias == 1)
        neuralNetwork.useHiddenBias = True;
    
    else
        return 5;
    
    if (i == 1)
        neuralNetwork.useMomentum = True;
    else if (i == 0)
        neuralNetwork.useMomentum = False;
    else
        return 6;
    
    return 0; /* All correct */
}

int netFileParser(FILE *fd)
{
    networkPrecision tmp = 0.0;
    int t = 0;
    int i,j,k;
    
    for (i=0;i<(neuralNetwork.inputNeurons + neuralNetwork.inputBias);i++)
    {
        for (j=0;j<(neuralNetwork.hiddenNeurons);j++)
        {
            if (fscanf(fd,"%lf\n", &tmp) < 0)
                return 1;
            neuralNetwork.inputWeights[i][j] = tmp;
            t++;
        }
    }
    
    if (neuralNetwork.hiddenLayers > 1)
    {
        for (k=0;k<(neuralNetwork.hiddenLayers - 1);k++)
        {
            for (i=0;i<(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias);i++)
            {
                for (j=0;j<(neuralNetwork.hiddenNeurons);j++)
                {
                    if (fscanf(fd,"%lf\n", &tmp) < 0)
                        return 2;
                    neuralNetwork.hiddenWeights[k][i][j] = tmp;
                    t++;
                }
            }
        }
    }
    
    for (i=0;i<(neuralNetwork.hiddenNeurons + neuralNetwork.hiddenBias);i++)
    {
        for (j=0;j<(neuralNetwork.outputNeurons);j++)
        {
            if (fscanf(fd,"%lf\n", &tmp) < 0)
                return 3;
            neuralNetwork.outputWeights[i][j] = tmp;
            t++;
        }
    }
    
    /* Missing checking of t value */
    return 0;
}






