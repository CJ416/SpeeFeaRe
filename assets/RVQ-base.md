## Pioneer work --- SoundStream (Google)

### 1. Overview

[SoundStream](https://arxiv.org/pdf/2107.03312) was proposed in 2021, it pioneered the neural audio codec, and most subsequent work has followed the **encoder-quantizer-decoder** network architecture. A single model can operate across variable bitrates from **3 kbps to 18 kbps**, with a negligible quality loss when compared with models trained at fixed bitrates.

![](./pics/soundstream.png 'SoundStream Architecture')

### 2. Core Components

Soundstream is a fully convolution-based AutoEncoder with fixed codebook quantizer. There were several notable tricks. 

* To support streamable inference, it adopted causal padding and causal convolution. 

* To improve the codebook usage, it adopt K-means on the first batch.

* When a codebook vector has not been assigned any input frame for several batches, we replace it with an input frame randomly sampled within the current
  batch



#### 2.1 Encoder and Decoder

![](/Users/jaykeecao/Documents/code/SpeeFeaRe/assets/pics/ss-arch.png)

#### 2.2 K-Means Codebook Initialization

**Benefit**

> Although the initial embeddings of a random encoder are noisier, and the cluster centers obtained by K-means are also biased towards randomness, this initialization is still useful: 
> 
> * it breaks the isotropic symmetry of the codebook (avoiding all codewords starting with the same initial values), reducing the problem of "dead codewords" or all samples mapping to the same few codewords in the early stages of training, thus making training more stable and converging faster.
> 
> * Initializing with a large amount of real input (rather than a single small batch) allows the clustering to reflect the data distribution, resulting in significantly better performance.

**Limitations**

>  If only a very small or unrepresentative batch is used, the initialization will be poor and may even interfere with training.

>  Using a **pre-trained encoder** (or performing K-means first using traditional audio features such as log-mel) will yield better results.

```python
def kmeans_init_codebook(self, data):
        '''
        K-means initialization of the codebook embeddings

        Args:
        -----
            data: Tensor[N, D]
                Data to use for k-means initialization
        '''
        if self.initted:
            return 
        N = data.shape[0]
        if N < self.codebook_size:
            indices = torch.randint(0, N, (self.codebook_size,))
            data = torch.cat([data, data[indices]], dim=0)
            N = data.shape[0]
        
        # 1. Randomly select initial centroids
        indices = torch.randperm(N)[:self.codebook_size]
        centroids = data[indices].clone()

        for iteration in range(self.kmeans_iters):
            distance = torch.cdist(data, centroids)     # N, K
            assignments = distance.argmin(dim=1)        # N

            new_centroids = centroids.clone()
            for k in range(self.codebook_size):
                mask = assignments == k                 # data points assigned to cluster k
                if mask.sum() > 0:
                    new_centroids[k] = data[mask].mean(dim=0)
                else:
                    random_idx = torch.randint(0, N, (1,))
                    new_centroids[k] = data[random_idx]
            
            diff = (new_centroids - centroids).pow(2).sum()
            centroids = new_centroids

            if iteration % 10 == 0:
                print(f'K-means init iteration {iteration}/{self.kmeans_iters}, diff: {diff:.6f}')
            
            if diff < 1e-6:
                print(f' K-means converged at {iteration}')
                break
        self.codebook.weight.data.copy_(centroids)
        self.initted.data.copy_(torch.tensor([True]))
```



## DAC (Descript-inc)

>  The DAC largely inherits the architectural ideas of SoundStream, and has made improvements based on the architecture.

### 1. [Periodic activation function](https://arxiv.org/abs/2006.08195)

$$
\text{Snake}_a:=x+\frac{1}{\alpha}\sin^2(\alpha x)
$$

Snake function is more suitable for periodic signals.

![](/Users/jaykeecao/Documents/code/SpeeFeaRe/assets/pics/snake%20activation.png)

**Sanke function key properties analysis**

****![](/Users/jaykeecao/Documents/code/SpeeFeaRe/assets/pics/snake_properties.png)



### 2. Comparison---DAC vs. SoundStream

|                   | SoundStream (2021)  | DAC (2023)                 |
| ----------------- | ------------------- | -------------------------- |
| **Activation**    | ELU/ReLU            | **Snake1d**                |
| **RVQ**           | Fixed codebook size | **dropout**                |
| **Discriminator** | Multiscale          | **MultiPeriod+MultiScale** |

[DAC---High-Fidelity Audio Compression with Imroved RVQGAN](https://arxiv.org/pdf/2306.06546)

There are also some useful traits mentioned in DAC to improve codebook usage.

* **Factorized codes**
  
  * Project input dim into codebook dim, reduce the vector dimension
  
  * >  Factorization decouples code lookup and code embedding, by performing code lookup in a low-dimensional space (8d or 32d) whereas the code embedding resides in a high dimensional space (1024d). Intuitively, this can be interpreted as a code lookup using only the principal components of the input vector that maximally explain the variance in the data.

* **L2-normalization**
  
  * > ****The L2-normalization of the encoded and codebook vectors converts euclidean distance to cosine similarity, which is helpful for stability and quality

* **Quantizer dropout**
  
  * > Apply quantizer dropout to each input example with some probability p.

Besides, DAC codec designs more complicated discriminators for more high-fidelity reconstruction.


