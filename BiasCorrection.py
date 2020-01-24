import numpy as np
from sklearn.linear_model import LinearRegression
from Error import mae, mse, rmse


class BiasCorrection:

    def score(self, obsrv, model, method="mae"):
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
        self.model = model
        self.obsrv = obsrv
        self.c = np.nanmean(self.model) - np.nanmean(self.obsrv)

    def bias_correction(self, model):
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
        self.model = model
        self.obsrv = obsrv
        self.k = 1 / (np.nanmean(self.model) / np.nanmean(self.obsrv))

    def bias_correction(self, model):
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
        self.model = model
        self.obsrv = obsrv
        lr = LinearRegression()
        lr.fit(self.model.reshape(-1, 1), self.obsrv)
        self.slope, self.intercept = np.round(lr.coef_[0], 6), np.round(lr.intercept_, 6)

    def bias_correction(self, model):
        corrected = model * self.slope + self.intercept
        return corrected