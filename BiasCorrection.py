import numpy as np
from sklearn.linear_model import LinearRegression
from .Error import mae, mse, rmse


class BiasCorrection:

    def score(self, obsrv, model, method="mae"):
        obsrv = np.array(obsrv, dtype=np.float)
        model = np.array(model, dtype=np.float)
        methods = {"mae": mae, "mse": mse, "rmse": rmse}
        error_cal = methods[method]
        corrected = self.bias_correction(model)
        return {
            f"{method}_before": error_cal(model, obsrv),
            f"{method}_after": error_cal(corrected, obsrv),
        }


class Shift(BiasCorrection):
    """
        bias coorection technique:
            model_corrected = model - c

            where c = mean(model) - mean(observed)
    """

    def fit(self, obsrv, model):
        """
            calculate c
        """
        self.model = np.array(model, dtype=np.float)
        self.obsrv = np.array(obsrv, dtype=np.float)

        
        if len(self.obsrv[~np.isnan(self.obsrv)]) > 0:
            self.c = np.nanmean(self.model) - np.nanmean(self.obsrv)
        else:
            # no not-nan value available in observed data
            print("Not enough observed data")
            self.c = 0

    def bias_correction(self, model):
        model = np.array(model, dtype=np.float) 
        corrected = model - self.c
        return corrected


class Scale(BiasCorrection):
    """
        bias correction technique:
            model_corrected = model * k

        where k = 1/(mean(model)/mean(observed))
    """

    def fit(self, obsrv, model):
        """
            calculate k
        """
        self.model = np.array(model, dtype=np.float)
        self.obsrv = np.array(obsrv, dtype=np.float)
        if len(self.obsrv[~np.isnan(self.obsrv)]) > 0:
            self.k = 1 / (np.nanmean(self.model) / np.nanmean(self.obsrv))
        else:
            # no not-nan value available in observed data
            print("Not enough observed data")
            self.k = 1


    def bias_correction(self, model):
        model = np.array(model, dtype=np.float) 
        corrected = model * self.k
        return corrected


class LinearReg(BiasCorrection):
    """
        bias correction technique:
            model_corrected = a*model + b
            # calculate from sklearn
    """

    def fit(self, obsrv, model):
        """
            calculate slope and intercept
        """
        self.model = np.array(model, dtype=np.float)
        self.obsrv = np.array(obsrv, dtype=np.float)
        if len(self.obsrv[~np.isnan(self.obsrv)]) > 0:
            lr = LinearRegression()
            lr.fit(self.model.reshape(-1, 1), self.obsrv)
            self.slope, self.intercept = np.round(lr.coef_[0], 6), np.round(lr.intercept_, 6)
        else:
            # no not-nan value available in observed data
            print("Not enough observed data")
            self.slope, self.intercept = 1, 0

    def bias_correction(self, model):
        model = np.array(model, dtype=np.float)
        corrected = model * self.slope + self.intercept
        return corrected