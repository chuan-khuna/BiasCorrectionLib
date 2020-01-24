# BiasCorrectionLib

ไลบรารี่สำหรับการปรับปรุงความเอนเอียงข้อมูลสภาพอากาศ

## Guide

```
/BiascorrectionLib
    |- Biascorrection.py
    |- Error.py
example.py
```

example.py

```python
from BiasCorrectionLib.BiasCorrection import Shift, Scale, LinearReg
import numpy as np

obs = np.arange(1, 11)
model = obs + 2     # model has bias

bc = Shift()
bc.fit(obs, model)              # train bias correction method
bc.score(obs, model, 'mae')     # calculate error
bc.bias_correction(model)       # return bias corrected model data
```
