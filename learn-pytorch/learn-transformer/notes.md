# Transformers

Lets get some vector representation of the sequence of tokens to give vectors $x_1, x_2, x_3$. Now pass each of these vectors through an encoder. Even though each sequence is encoded separately, the weights are shared, i.e., 
$$
h_1 = enc(W, x_1) \\
h_2 = enc(W, x_2) \\
h_3 = enc(W, x_3)
$$
The encoder can be the usual linear layer followed by some non-linearity, e.g., $enc(W, x_t) = \sigma(Wx_t + b)$. Now take the encoded sequence and generate values, keys, and queries, again using functions with shared weights.
$$
v_1 = v(W_v, h_1) \\
v_2 = v(W_v, h_2) \\
v_3 = v(W_v, h_3)
$$
Similarly for keys and queries -
$$
k_t = k(W_k, h_t) \\
q_t = q(W_q, h_t)
$$
<img src="./self_attn_1.png" alt="self_attn_1" style="zoom:70%;" />

The intuition behind doing this is that I want to mix the encoded vector at my current time step with the encoded vector from some other time step that is most relevant to mine. Queries represent the information that this time step is looking for, keys represent the information that this time step is willing to provide, and finally values are the actual encoded values. In order to mix the value of the most relevant time step with my value, I can find find out which key is closest to my query. I can do this by taking the dot product between my query and all other keys, also my own key, why not? $q_t^Tk_{t'}$ . The maximum dot product is for the most relevant time step, i.e., $\tau = argmax_{t'}(q_t^Tk_{t'})$. Now I can use this most relevant value $v_{\tau}$ however I want to. The problem is that $argmax$ is not a differentiable function. To workaround this, I can use the softmax 🤯. I can take a weighted sum of all the values, but the weights are the softmax of the dotproducts. Remember the most relevant dotproduct is supposed to have the highest value, therefore its softmax will be close to 1 and the others will be close to 0. This way I can pick off the most relevant value.

First calculate the weights for the weighted sum -
$$
e_{lt} = <q_l, k_t> \\
\alpha_{lt} = \frac{exp(e_{lt})}{\sum_{t'} exp(e_{lt'})} \\
$$
<img src="./self_attn_wts.png" alt="self_attn_weights" style="zoom:67%;" />



And then compute the weighted sum -
$$
a_l = \sum_t \alpha_{lt} \cdot v_t
$$
<img src="./self_attn_weighted_sum.png" alt="self_attn_weighted_sum" style="zoom:67%;" />



Here is the entire self-attention layer -

<img src="./self_attn_full.png" alt="self_attn_full" style="zoom:67%;" />

Simplifying -

![self_attn](./self_attn.png)