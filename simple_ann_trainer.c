#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//
//important variables to train the nueral network
//

float input[40][6]; //input data for expiratory phase(20) and inspiratory(20) and for the six categories of disease (this would remain unchanged)
float input_perturbed[40][6*500]; // this would contain all perturbed data

float weightih[40][30]; // weight between the 40 input data and the 30 hidden layer node
float weightih1[40][30];


float hiddenbias[1][30]; // bias weigths of the hidden layer 
float hiddenoutput[1][30]; // output of the hidden layer node

float weightho[30][6]; // weight between the 30 hidden layer node and the 6 output layer nodes
float weightho1[30][6];


float outputbias[1][6]; // bias weigths of the output layer  

float output[6][6]; // output of the output layer nodes
float idealoutput[6][6]; // ideal output we are meant to get



//
//important parameters to train the nueral network
//

int disease;
int sample;
float momentum;
float learningrate;




///////////////////////////////////////////////////
//Mersenne Twister random number generator starts//
///////////////////////////////////////////////////
/* 
   A C-program for MT19937-64 (2004/9/29 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.

   This is a 64-bit version of Mersenne Twister pseudorandom number
   generator.

   Before using, initialize the state by using init_genrand64(seed)  
   or init_by_array64(init_key, key_length).

   Copyright (C) 2004, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   References:
   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
     ACM Transactions on Modeling and 
     Computer Simulation 10. (2000) 348--357.
   M. Matsumoto and T. Nishimura,
     ``Mersenne Twister: a 623-dimensionally equidistributed
       uniform pseudorandom number generator''
     ACM Transactions on Modeling and 
     Computer Simulation 8. (Jan. 1998) 3--30.

   Any feedback is very welcome.
   http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
*/


#include <stdio.h>

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL /* Least significant 31 bits */


/* The array for the state vector */
static unsigned long long mt[NN]; 
/* mti==NN+1 means mt[NN] is not initialized */
static int mti=NN+1; 

/* initializes mt[NN] with a seed */
void init_genrand64(unsigned long long seed)
{
    mt[0] = seed;
    for (mti=1; mti<NN; mti++) 
        mt[mti] =  (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array64(unsigned long long init_key[],
		     unsigned long long key_length)
    {
    unsigned long long i, j, k;
    init_genrand64(19650218ULL);
    i=1; j=0;
    k = (NN>key_length ? NN : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * 3935559000370003845ULL))
          + init_key[j] + j; /* non linear */
        i++; j++;
        if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=NN-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * 2862933555777941757ULL))
          - i; /* non linear */
        i++;
        if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
    }

    mt[0] = 1ULL << 63; /* MSB is 1; assuring non-zero initial array */ 
    }

/* generates a random number on [0, 2^64-1]-interval */
unsigned long long genrand64_int64(void)
    {
    int i;
    unsigned long long x;
    static unsigned long long mag01[2]={0ULL, MATRIX_A};

    if (mti >= NN) { /* generate NN words at one time */

        /* if init_genrand64() has not been called, */
        /* a default initial seed is used     */
        if (mti == NN+1) 
            init_genrand64(5489ULL); 

        for (i=0;i<NN-MM;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+MM] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        for (;i<NN-1;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+(MM-NN)] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        x = (mt[NN-1]&UM)|(mt[0]&LM);
        mt[NN-1] = mt[MM-1] ^ (x>>1) ^ mag01[(int)(x&1ULL)];

        mti = 0;
    }
  
    x = mt[mti++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
    }

/* generates a random number on [0, 2^63-1]-interval */
long long genrand64_int63(void)
    {
    return (long long)(genrand64_int64() >> 1);
    }

/* generates a random number on [0,1]-real-interval */
double genrand64_real1(void)
    {
    return (genrand64_int64() >> 11) * (1.0/9007199254740991.0);
    }

/* generates a random number on [0,1)-real-interval */
double genrand64_real2(void)
    {
    return (genrand64_int64() >> 11) * (1.0/9007199254740992.0);
    }

/* generates a random number on (0,1)-real-interval */
double genrand64_real3(void)
    {
    return ((genrand64_int64() >> 12) + 0.5) * (1.0/4503599627370496.0);
    }

////////////////////////////////////////////////
//Mersenne Twister random number generator end//
////////////////////////////////////////////////



int random()
    {
    int h,i,j;
    unsigned long long init[4]={0x12345ULL, 0x23456ULL, 0x34567ULL, 0x45678ULL}, length=4;
    init_by_array64(init, length);
    
    float temp;
    for(h=0;h<500;h++)
           
        {
        for (i=0;i<6;i++)/// input layer array
            {
            for (j=0;j<40;j++)
                {
                temp=(genrand64_real2()*0.1)-0.05;//  -/+5%
                              
                              
                input_perturbed[j][i+(h*6)]=(input[j][i]*(1+temp));
                //printf("%f  %f %f ",input1[j][i*(h+1)], 1+temp,input[j][i]);
                };
            //printf("\n\n");
                            
                            
            }
          }        
     
    }



//sets up all the arrrays
int setup()
    {
    FILE *pvalue;
    FILE *pweightih;
    FILE *pweightho;
    FILE *phiddenbias;
    FILE *poutputbias;

    // file with the respiratory datas
    pvalue = fopen("data//respiratory_data.txt", "r");
    // file that stores the values of the weight between input layer and output layer
    pweightih = fopen("initial_model//input_hidden_weights.txt","r");
    pweightho = fopen("initial_model//hidden_output_weights.txt","r");
    phiddenbias = fopen("initial_model//hidden_bias_weights.txt","r");
    poutputbias = fopen("initial_model//output_bias_weights.txt","r");


    int i, j;
          
          
    if (pvalue!=NULL||pweightih!=NULL||pweightho!=NULL||phiddenbias!=NULL||poutputbias!=NULL)
        {
        // input layer array
        for (i = 0; i < 6; i++)
            {
            for (j = 0; j < 40; j++)
                {
                fscanf(pvalue, "%f ", &input[j][i]);                
                }
            }

        for (i = 0; i < 30; i++) //weigth between input and hidden layer
            {
                         
            for (j = 0; j < 40; j++)
                {
                fscanf(pweightih, "%f ", &weightih[j][i]);
                weightih1[j][i] = weightih[j][i];
                }
            }
             
             
             
        //weigth between hiddenlayer bias and the hiddenlayer nodes                      
        for (j = 0;j < 30; j++)
            {
            fscanf(phiddenbias, "%f ", &hiddenbias[0][j]);                    
            }

        //weigth between ouputlayer bias and the outputlayer nodes                         
        for (j=0;j<6;j++)
            {
            fscanf(poutputbias, "%f ",&outputbias[0][j]);
            }  
                        
                        
        //weigth between hidden and output layer                 
        for (i = 0; i < 6; i++) 
            {        
            for (j = 0; j < 30; j++)
                {             
                fscanf(pweightho, "%f ", &weightho[j][i]);
                weightho1[j][i] = weightho[j][i];
                }
            }     
        }
    else
        printf("cannot open");
        
    
           
           
    // output layer array
    idealoutput[6][6] = 0;
    for (i = 0; i < 6; i++)
        {   
        idealoutput[i][i]=1;
        }
         
    fclose(pvalue);
    fclose(pweightih);
    fclose(pweightho);
    fclose(phiddenbias);
    fclose(poutputbias);
         
    }


int feedforward()
    {
    // hidden layer update
    int i,j;
                  
    for (i = 0; i < 30; i++)
        {
        float total = 0;
        for (j = 0; j < 40; j++)
            {
            float temp;
            temp = input_perturbed[j][disease + sample * 6] * weightih[j][i];
            total = total + temp;
            }
                          
        total = total + hiddenbias[0][i];      
        hiddenoutput[0][i] = 1/(1 + exp(-1 * total));
        }             

    // output layer update      
    for (i = 0;i < 6; i++)
        {
        float total = 0;
        for (j = 0; j < 30; j++)
                {
          
                float temp;
                temp= hiddenoutput[0][j]*weightho[j][i];
                                          
                total=total+temp;
                };

        total = total + outputbias[0][i];       
        output[i][disease] = 1/(1 + exp(-1 * total));
        // printf(" %f ", output[i][disease]);
        // printf("\n");
        }         
    }
  
   
int backpropagation()
    {
    int i, j, k,l;
            
    for (i = 0; i < 6; i++) 
        {                
        // weight between input and hidden layer
        for (j = 0; j < 30; j++)
            {  
            float dweightho, errorweightrate, deltak; 
                                        
            deltak = output[i][disease] * (1 - output[i][disease]) * (output[i][disease] - idealoutput[i][disease]);  
            errorweightrate = deltak * hiddenoutput[0][j];
            dweightho = -1 * learningrate * errorweightrate;
            // printf(" %f weight %f ", dweightho, weightho[j][i]);  
            weightho[j][i] = weightho[j][i] + dweightho;
            }
        // printf("\n\n");    
        }
           
    for (i = 0; i < 30; i++) 
        {
        // weight between input and hidden layer
        for (j = 0; j < 40; j++)
                {  
                float dweightih, errorweightrate, deltakweighthosum=0,deltaj;   
                              
                for (k = 0; k < 6; k++)
                    {
                    float deltak;
                                        
                    deltak=output[k][disease] * (1-output[k][disease]) * (output[k][disease] - idealoutput[k][disease]);
                    deltakweighthosum = deltakweighthosum + (weightho[i][k] * deltak);  
                    } 
                deltaj = hiddenoutput[0][i] * (1 - hiddenoutput[0][i]) * deltakweighthosum;
                              
                errorweightrate = deltaj * input_perturbed[j][disease + (sample*6)];

                dweightih = -1 * learningrate * errorweightrate;    
                // printf(" %f we%f ",dweightih, weightih[j][i]);   
                weightih[j][i] = weightih[j][i] + dweightih ;
                // printf(" %f ", weightih[j][i]);             
                }
            // printf("\n\n");         
        }           
                        
    }                


int storeweight()
    {
    FILE *pvalue;
    FILE *pweightih;
    FILE *pweightho;
    FILE *phiddenbias;
    FILE *poutputbias;
    
          
    pweightih = fopen("model//input_hidden_weights.txt","w");
    pweightho = fopen("model//hidden_output_weights.txt","w");
    phiddenbias = fopen("model//hidden_bias_weights.txt", "w");
    poutputbias = fopen("model//output_bias_weights.txt", "w");
    int i, j;
          
    if (pvalue!=NULL || pweightih != NULL || pweightho != NULL || phiddenbias != NULL || poutputbias != NULL)
        {
        // weight between input and hidden layer
        for (i = 0; i < 30; i++) 
            {
                         
            for (j = 0;j < 40; j++)
                {
                fprintf(pweightih, "%f\t", weightih[j][i]);
                }
            fprintf(pweightih, "\n " );     
            }

        // weight between hiddenlayer bias and the hiddenlayer nodes
        for (j = 0; j < 30; j++) 
            {
            fprintf(phiddenbias, "%f\t", hiddenbias[0][j]);                    
            }

        // weight between ouputlayer bias and the outputlayer nodes
        for (j = 0; j < 6; j++) 
            {
            fprintf(poutputbias, "%f\t", outputbias[0][j]);
            }  

        // weight between hidden and output layer                
        for (i = 0; i < 6; i++)  
            {   
            for (j=0;j<30;j++)
                {
                fprintf(pweightho, "%f\t", weightho[j][i]);
                };
            fprintf(pweightho, "\n ");     
            }
        }
   else
        {
        printf("Failed to open files!");
        }
            
          
    fclose(pweightih);
    fclose(pweightho);
    fclose(phiddenbias);
    fclose(poutputbias);           
    }


int main()
    {
    disease = 0;
    learningrate = 0.07;
    momentum = 0.3;
    sample = 7;
    setup();
    int j;
    
    random(); 
    feedforward();
    
   /*
    for (j = 0; j < 6; j++)
        {
        printf("%f  %f \n", output[j][1], idealoutput[j][1]);
    
        } 
    */
      
    //number of iteration    
    for (j = 0; j < 100; j++)
        { 
        for (disease = 0; disease < 6; disease++)
            {
            // number of sample datas to use out of a possible 500
            for (sample=0; sample < 200; sample++)
                {
                feedforward();
                backpropagation(); 
                feedforward();
                }     
            }
        }

    feedforward();
    printf("\n\n\n");

    for (sample = 250; sample < 251; sample++) 
        {
        for (j = 0; j < 6; j++)
            {
            printf("%f  %f \n",output[j][1], idealoutput[j][1]);
            }
        printf("\n\n");        
        } 
    
    storeweight();
    system("PAUSE");
    }

