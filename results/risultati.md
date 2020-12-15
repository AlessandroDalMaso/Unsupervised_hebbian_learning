## 12/12/2020

### 1

con il mio codice ho fatto 160 epochs con batch a 100.

![](12-12-2020/1/2020-12-12-weights.png)

![](12-12-2020/1/2020-12-12-norms.png)

![](12-12-2020/1/2020-12-12-ravel.png)

## 15/12/2020

### 1
con il codice originale, 160 batch da 50 samples

![](15-12-2020/1/15-12-2020-original.png)

### 2

ho splittato mnist 25% di test e ho fittato una random forest, senza preprocessare. risultato: score = 0.97

### 3

ho fatto un fit con mnist 25% di test 160 epochs batch da 99 e ho fittato una random forest con i risultati. risultato: score = 0.943
![](15-12-2020/2/p-norms.png)

![](15-12-2020/2/weights_heatmap.png)

![](15-12-2020/2/weights_unraveled.png)
