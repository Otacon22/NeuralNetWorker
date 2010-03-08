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

typedef double networkPrecision;
typedef double errorPrecision;
typedef enum {False, True} boolean;

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

/* Functions prototypes area */
networkPrecision randomGenerator(double, double);
networkPrecision randomizer();
networkPrecision randomizer2();
networkPrecision transferFunction_Sigmodial(networkPrecision);
networkPrecision transferFunction_SigmodialDerived(networkPrecision);
networkPrecision transferFunction_Step(networkPrecision);
networkPrecision transferFunction_StepDerived(networkPrecision);
networkPrecision transferFunction_Tanh(networkPrecision);
networkPrecision transferFunction_TanhDerived(networkPrecision);
void weightsDeallocatorInput(int);
void weightsDeallocatorInputMomentum(int);
void weightsDeallocatorHidden(int, int);
void weightsDeallocatorHiddenMomentum(int, int);
void weightsDeallocatorOutput(int);
void weightsDeallocatorOutputMomentum(int);
void weightsDeallocator();
void activationDeallocatorInput();
void activationDeallocatorHidden(int);
void activationDeallocatorOutput();
void activationDeallocator();
int activationAllocator();
void setBias();
void flushWeights();
void updateNetwork();
void deltaDeallocatorHidden(int);
void deltaDeallocatorOutput();
void deltaDeallocator();
int deltaAllocator();
errorPrecision backPropagation(float, float, int);
errorPrecision trainNetwork(float, float, int, errorPrecision);