## Audio Codec Tutorial --- SpeeFeaRe

Speech Feature Representation repo, containing regular continuous and discrete speech representations. 

### To-do-list

* Introduce the components of neural audio codec 
* Regular neural audio codec techniques will be added into the repository. 
* Based on the neural audio codec, build a LLM-based TTS system from scratch



#### DAC

##### 1. Model Architecture

DAC is a pure convolution-based and standard encoder-quantizer-decoder architecture. As mentioned above, DAC neural audio codec contains 3 components: encoder, quantizer, and decoder.

###### Encoder and Decoder---information compression and reconstruction



###### Quantizer---discrete the continuous representation



##### 2. Training tricks and paradigm

###### Factorize the quantized code

###### Quantizer Dropout

###### Adversarial training

###### ...

##### 3. Training Scripts

> We can train the dac model using the following scripts.

```shell
cd scripts
python train_dac.py
```


