# Librerias de Python
import pandas as pd
import numpy as np
# Librerias de analisis de datos y estadistica
import statsmodels.api as sm
from scipy.stats import f
from scipy.stats import norm
from statsmodels.tsa.stattools import zivot_andrews
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from scipy.special import expit  # sigmoid function
# Librerias de analisis de series temporales
import ruptures as rpt
# Librerias de visualizacion
import matplotlib.pyplot as plt


class Chow_SB:
    """
    Una clase para analizar puntos de quiebre estructural en series temporales.
    """

    def __init__(self, df_plano: pd.DataFrame,df_change_time: pd.DataFrame):
        """
        Inicializa la clase con las rutas a los archivos de datos.

        Args:
            x_train_file (str): La ruta al archivo parquet de características (X_train).
            y_train_file (str): La ruta al archivo parquet de etiquetas (y_train).
        """

        self.df_plano = df_plano
        self.df_change_time = df_change_time

    def perform_chow_test(self, selected_id, series_used='value', k=2, alpha=0.05):
        """
        Realiza el test de Chow para un ID seleccionado.

        Args:
            selected_id (int): El ID de la serie temporal a analizar.
            series_used (str): El nombre de la columna de la serie a usar.
            k (int): Número de parámetros en el modelo.
            alpha (float): El nivel de significancia.

        Returns:
            dict: Un diccionario con los resultados del test de Chow.
        """
        if self.df_plano is None:
            return {"error": "Los datos no han sido cargados."}

        df_plano_id = self.df_plano[self.df_plano['id'] == selected_id].copy()
        df_plano_id.rename(columns={'time': 'Index'}, inplace=True)
         
        df_plano_antes = df_plano_id[df_plano_id['period'] == 0]
        df_plano_despues = df_plano_id[df_plano_id['period'] == 1]

        # Ajuste de modelos
        X_full = sm.add_constant(df_plano_id['Index'])
        X1 = sm.add_constant(df_plano_antes['Index'])
        X2 = sm.add_constant(df_plano_despues['Index'])

        model_full = sm.OLS(df_plano_id[series_used], X_full).fit()
        model1 = sm.OLS(df_plano_antes[series_used], X1).fit()
        model2 = sm.OLS(df_plano_despues[series_used], X2).fit()

        # Suma de residuos al cuadrado
        ssr_full = model_full.ssr
        ssr1 = model1.ssr
        ssr2 = model2.ssr

        # Numero de observaciones en cada segmento
        n = len(df_plano_id)
        n1 = len(df_plano_antes)
        n2 = len(df_plano_despues)

        # Estadístico de Chow
        chow_stat = ((ssr_full - (ssr1 + ssr2)) / k) / ((ssr1 + ssr2) / (n1 + n2 - 2 * k))
        p_value = 1 - f.cdf(chow_stat, dfn=k, dfd=n1 + n2 - 2 * k)
        f_critical = f.ppf(1 - alpha, dfn=k, dfd=n1 + n2 - 2 * k)

        break_detected = chow_stat > f_critical

        results = {
            "chow_statistic": chow_stat,
            "p_value": p_value,
            "f_critical_value": f_critical,
            "break_detected": break_detected,
            "message": "Punto de quiebre estructural detectado." if break_detected else "No se detectó un punto de quiebre estructural significativo."
        }
        return results

    def plot_structural_break(self, selected_id, series_used='value'):
        """
        Visualiza la serie temporal y el punto de quiebre estructural.

        Args:
            selected_id (int): El ID de la serie temporal a visualizar.
            series_used (str): La columna de la serie a usar.
        """
        if self.df_plano is None or self.df_change_time is None:
            print("Los datos no han sido cargados.")
            return

        df_plano_id = self.df_plano[self.df_plano['id'] == selected_id].copy()
        df_plano_id.rename(columns={'time': 'Index'}, inplace=True)
        
        
        try:
            structural_break_time = self.df_change_time[self.df_change_time['id'] == selected_id]['min_time_period_1'].values[0]
        except IndexError:
            print(f"No se encontró un punto de quiebre para el id {selected_id}")
            return


        plt.figure(figsize=(12, 7))
        plt.plot(df_plano_id['Index'], df_plano_id[series_used], label=f'{series_used.replace("_", " ").title()} Price')
        plt.axvline(x=structural_break_time, color='red', linestyle='--', label=f'Punto de quiebre ({structural_break_time})')
        plt.title(f'Análisis de Punto de Quiebre para ID: {selected_id}')
        plt.xlabel('Tiempo')
        plt.ylabel(f'Valor de {series_used.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True)
        plt.show()

class ZivotAndrewsTest:
    """
    Una clase para realizar el test de Zivot-Andrews, mostrar los resultados
    y visualizar el punto de quiebre estructural en una serie temporal.
    """

    def __init__(self, dataframe: pd.DataFrame, value_column: str, date_column: str = None):
        """
        Inicializa la clase con el DataFrame y los nombres de las columnas.

        Args:
            dataframe (pd.DataFrame): El DataFrame que contiene los datos.
                                      Debe tener un índice de tipo fecha (DatetimeIndex).
            value_column (str): El nombre de la columna con la serie de tiempo a analizar.
            date_column (str, optional): El nombre de la columna de fecha para mostrar en el gráfico.
                                         Si es None, se usará el índice.
        """
        
        # renombramos la columna 'time' a 'Index' si existe
        if 'time' in dataframe.columns:
            dataframe = dataframe.rename(columns={'time': 'Index'})

        
        self.df = dataframe
        self.value_column = value_column
        self.date_column = date_column if date_column else dataframe.index.name
        self.result = None
        self.break_date = None
        self.break_date_index = None

    def run_test(self,selected_id: int, maxlag: int = 30, regression: str = 'ct'):
        """
        Ejecuta el test de Zivot-Andrews en la serie de datos.

        Args:
            selected_id (int): El ID de la serie temporal a analizar.
            maxlag (int): El número máximo de lags a considerar.
            regression (str): El tipo de regresión a utilizar ('c', 't', 'ct').
                              'c': constante
                              't': tendencia
                              'ct': constante y tendencia
        """
        print("Ejecutando el test de Zivot-Andrews...")
        # Filtrar el DataFrame por el ID seleccionado
        self.df = self.df[self.df['id'] == selected_id]
        # Asegurarse de que la columna de valores no tenga NaN
        series = self.df[self.value_column].dropna()
        za_result = zivot_andrews(series, maxlag=maxlag, regression=regression)
        
        self.result = za_result
        self.break_date_index = za_result[4]
        self.break_date = self.break_date_index
        print("Test completado.")

    def display_results(self):
        """
        Muestra los resultados del test de Zivot-Andrews de forma legible.
        """
        if self.result is None:
            print("Error: Debes ejecutar el test primero con el método 'run_test()'.")
            return
            
        print("\n--- Resultados del Test de Zivot-Andrews ---")
        print(f"Estadístico de prueba:   {self.result[0]:.4f}")
        print(f"P-valor:                   {self.result[1]:.4f}")
        print("Valores Críticos:")
        for key, value in self.result[2].items():
            print(f"  {key}: {value:.4f}")
        print(f"Número de Lags Usados:     {self.result[3]}")
        print(f"Índice del Punto de Quiebre: {self.break_date_index}")
        print(f"Fecha del Punto de Quiebre: {self.break_date}")
        print("------------------------------------------")

    def plot_breakpoint(self, title='Serie con Punto de Quiebre de Zivot-Andrews', 
                        xlabel='Fecha', ylabel='Valor'):
        """
        Genera un gráfico de la serie temporal mostrando el punto de quiebre identificado.
        """
        if self.result is None:
            print("Error: Debes ejecutar el test primero con el método 'run_test()'.")
            return

        plt.figure(figsize=(12, 7))
        plt.plot(self.df.Index, self.df[self.value_column], label=self.value_column)
        
        # Etiqueta para la línea vertical con la fecha del quiebre
        break_label = f"Punto de Quiebre ({self.break_date})"
        plt.axvline(x=self.break_date, color='red', linestyle='--', label=break_label)
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

class ZivotAndrewsFixedBreakTest:
    """
    Realiza una prueba de quiebre estructural en un punto fijo específico
    usando una regresión con dummies para capturar el quiebre.
    """

    def __init__(self, dataframe: pd.DataFrame, value_column: str, date_column: str = None, message: bool = True):
        if 'time' in dataframe.columns:
            dataframe = dataframe.rename(columns={'time': 'Index'})
        
        self.df = dataframe.copy()
        self.df_filtered = None
        self.value_column = value_column
        self.date_column = date_column if date_column else dataframe.index.name
        self.result = None
        self.break_date = None
        self.break_index = None
        self.message = message

    def run_test(self, selected_id: int, break_index: int, regression: str = 'ct', lags: int = 0):
        """
        Ejecuta el test con quiebre en un punto fijo usando regresión con dummies.

        Args:
            selected_id (int): ID de la serie a analizar.
            break_index (int): Índice (entero) del punto de quiebre a evaluar.
            regression (str): 'c', 't' o 'ct' para constante, tendencia o ambas.
            lags (int): Número de lags autorregresivos (no implementado en esta versión simplificada).
        """
        if self.message:
            print("Ejecutando el test con punto de quiebre fijo...")

        self.df_filtered = self.df[self.df['id'] == selected_id].reset_index(drop=True)
        y = self.df_filtered[self.value_column].dropna()
        t = np.arange(len(y))

        # Variables explicativas
        X = pd.DataFrame(index=y.index)
        if regression in ('c', 'ct'):
            X['const'] = 1
        if regression in ('t', 'ct'):
            X['trend'] = t

        # Dummy post-quiebre y shift de nivel y tendencia
        X['DU'] = (t >= break_index).astype(int)
        if regression in ('t', 'ct'):
            X['DT'] = X['DU'] * (t - break_index)

        # Regresión
        model = sm.OLS(y, X).fit()
        self.result = model
        self.break_index = break_index
        self.break_date = self.df_filtered.iloc[break_index]['Index']
        if self.message:
            print("Test completado.")

    def display_results(self):
        """
        Muestra los resultados de la regresión con quiebre estructural en punto fijo.
        """
        if self.result is None:
            print("Error: Debes ejecutar el test primero con el método 'run_test()'.")
            return
        
        print("\n--- Resultados del Test con Punto Fijo ---")
        print(self.result.summary())
        print(f"\nPunto de quiebre fijado en el índice: {self.break_index}")
        print(f"Fecha correspondiente: {self.break_date}")
        print("------------------------------------------")

    def plot_breakpoint(self, title='Serie con Punto de Quiebre Fijo', 
                        xlabel='Fecha', ylabel='Valor'):
        """
        Genera un gráfico de la serie temporal mostrando el punto de quiebre.
        """
        if self.result is None:
            print("Error: Debes ejecutar el test primero con el método 'run_test()'.")
            return

        plt.figure(figsize=(12, 7))
        plt.plot(self.df_filtered['Index'], self.df_filtered[self.value_column], label=self.value_column)
        break_label = f"Punto de Quiebre Fijo ({self.break_date})"
        plt.axvline(x=self.break_date, color='red', linestyle='--', label=break_label)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


class Ruptures:
    """
    Clase para detectar puntos de quiebre estructural en series temporales utilizando
    diferentes algoritmos del paquete ruptures.

    Métodos disponibles:
        - run_dynp: Algoritmo Dynamic Programming.
        - run_pelt: Algoritmo PELT.
        - run_binseg: Algoritmo Binary Segmentation.
        - run_window: Algoritmo Window-based.
        - run_bottomup: Algoritmo Bottom-Up.
        - run_kernel: Algoritmo Kernel-based con diferentes kernels.
        - run_all_models: Ejecuta todos los métodos anteriores secuencialmente.

    Args:
        df_plano (pd.DataFrame): DataFrame con los datos de la serie temporal.
        selected_id (int): ID de la serie a analizar.
        min_size (int): Tamaño mínimo de segmento para los algoritmos.
    """

    def __init__(self, df_plano, selected_id, min_size=28):
        self.df = df_plano[df_plano['id'] == selected_id]
        self.signal = self.df['value'].values
        self.dates = self.df['time']
        self.min_size = min_size

    def run_dynp(self, n_bkps=1, plot=True):
        algo = rpt.Dynp(model="l1", min_size=self.min_size)
        algo.fit(self.signal)
        bkps = algo.predict(n_bkps=n_bkps)

        if plot:
            rpt.display(self.signal, bkps, figsize=(10, 6))
            plt.show()

            plt.figure(figsize=(12, 6))
            plt.plot(self.dates, self.signal, label='Signal', color='blue')
            for bkp in bkps[:-1]:  # Excluir el último punto artificial
                plt.axvline(x=self.dates.iloc[bkp], color='red', linestyle='--', label='Detected Breakpoint')
            plt.title('Detected Breakpoint in Signal - Dynp')
            plt.xlabel('Date')
            plt.ylabel('Signal Value')
            plt.legend()
            plt.show()

        return bkps

    def run_pelt(self, pen=1, plot=True):
        algo = rpt.Pelt(model="l1", min_size=self.min_size)
        algo.fit(self.signal)
        result = algo.predict(pen=pen)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.dates, self.signal)
            for bkp in result[:-1]:
                ax.axvline(x=self.dates.iloc[bkp - 1], color='k', linestyle='--')
            ax.set_title("Pelt model")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

        return result

    def run_binseg(self, pen=1, plot=True):
        algo = rpt.Binseg(model="l2", min_size=self.min_size)
        algo.fit(self.signal)
        result = algo.predict(pen=pen)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.dates, self.signal)
            for bkp in result[:-1]:
                ax.axvline(x=self.dates.iloc[bkp - 1], color='k', linestyle='--')
            ax.set_title("Binseg model")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

        return result

    def run_window(self, pen=1, plot=True):
        algo = rpt.Window(model="l2", width=self.min_size)
        algo.fit(self.signal)
        result = algo.predict(pen=pen)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.dates, self.signal)
            for bkp in result[:-1]:
                ax.axvline(x=self.dates.iloc[bkp - 1], color='k', linestyle='--')
            ax.set_title("Window model")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

        return result

    def run_bottomup(self, pen=1, plot=True):
        algo = rpt.BottomUp(model="l2", min_size=self.min_size)
        algo.fit(self.signal)
        result = algo.predict(pen=pen)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(self.dates, self.signal)
            for bkp in result[:-1]:
                ax.axvline(x=self.dates.iloc[bkp - 1], color='k', linestyle='--')
            ax.set_title("BottomUp model")
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

        return result

    def run_kernel(self, n_bkps=4, plot=True):
        kernels = ['linear', 'rbf', 'cosine']
        results = {}

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, kernel in enumerate(kernels):
            algo = rpt.KernelCPD(kernel=kernel, min_size=self.min_size)
            algo.fit(self.signal)
            result = algo.predict(n_bkps=n_bkps)
            results[kernel] = result

            if plot:
                axes[i].plot(self.signal)
                for bkp in result[:-1]:
                    axes[i].axvline(x=bkp, color='k', linestyle='--')
                axes[i].set_title(f"Kernel model with {kernel} kernel")

        if plot:
            fig.tight_layout()
            plt.show()

        return results

    def run_all_models(self, dynp_nbkps=1, kernel_nbkps=4, pen=1, plot=True):
        """
        Ejecuta todos los algoritmos de detección de rupturas implementados en la clase.

        Returns:
            dict: Diccionario con los resultados de cada método.
        """
        return {
            "dynp": self.run_dynp(n_bkps=dynp_nbkps, plot=plot),
            "pelt": self.run_pelt(pen=pen, plot=plot),
            "binseg": self.run_binseg(pen=pen, plot=plot),
            "window": self.run_window(pen=pen, plot=plot),
            "bottomup": self.run_bottomup(pen=pen, plot=plot),
            "kernel": self.run_kernel(n_bkps=kernel_nbkps, plot=plot)
        }

class IMIModel:
    """
    IMIModel detects structural breaks (regimes) in a univariate time series using Gaussian Mixture Models (GMM) and model selection via the Integrated Completed Likelihood (ICL) criterion.
    Attributes:
        y (np.ndarray): The input univariate time series data.
        T (int): Length of the time series.
        max_regimes (int): Maximum number of regimes (components) to consider.
        random_state (np.random.RandomState): Random state for reproducibility.
        best_model (GaussianMixture): The best fitted GMM model according to the ICL criterion.
        regime_sequence (np.ndarray): Sequence of detected regimes for each time point.
        breakpoints (np.ndarray): Indices where regime changes (breakpoints) are detected.
    Methods:
        fit():
            Fits GMMs with 1 to max_regimes components, selects the best model using the ICL criterion, and determines the regime sequence and breakpoints.
        plot_regimes():
            Plots the time series colored by detected regimes and marks the detected breakpoints.
    """
    def __init__(self, y, max_regimes=5, seed=42):
        self.y = y
        self.T = len(y)
        self.max_regimes = max_regimes
        self.random_state = np.random.RandomState(seed)
        self.best_model = None
        self.regime_sequence = None
        self.breakpoints = None

    def fit(self):
        best_icl = np.inf
        best_result = None

        for n_regimes in range(1, self.max_regimes + 1):
            gmm = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=self.random_state)
            y_reshaped = self.y.reshape(-1, 1)
            gmm.fit(y_reshaped)

            log_likelihood = gmm.score(y_reshaped) * self.T
            responsibilities = gmm.predict_proba(y_reshaped)
            classification_entropy = -np.sum(responsibilities * np.log(responsibilities + 1e-10))
            n_params = n_regimes - 1 + n_regimes + n_regimes  # weights, means, variances
            icl_bic = -2 * log_likelihood + np.log(self.T) * n_params + 2 * classification_entropy

            if icl_bic < best_icl:
                best_icl = icl_bic
                best_result = {
                    "model": gmm,
                    "n_regimes": n_regimes,
                    "responsibilities": responsibilities
                }

        self.best_model = best_result["model"]
        self.regime_sequence = np.argmax(best_result["responsibilities"], axis=1)
        self.breakpoints = np.where(np.diff(self.regime_sequence) != 0)[0] + 1

    def plot_regimes(self):
        plt.figure(figsize=(12, 4))
        for i in range(self.best_model.n_components):
            plt.plot(np.where(self.regime_sequence == i)[0],
                     self.y[self.regime_sequence == i],
                     '.', label=f'Regime {i}')
        for bp in self.breakpoints:
            plt.axvline(bp, color='black', linestyle='--', alpha=0.6)
        plt.title("Detected Regimes and Breakpoints")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

class FuzzyChowBreakDetector:
    """
    Detects structural breaks in a time series using a fuzzy version of the Chow test.
    This approach uses fuzzy partitioning of the time axis and computes local linear trends
    (via fuzzy transform) to identify points where the trend changes sharply, indicating a possible break.

    Attributes:
        time (np.ndarray): Array of time indices.
        values (np.ndarray): Array of observed values.
        n_nodes (int): Number of fuzzy nodes (partitions) to use.
        small_threshold (float): Threshold for detecting near-zero slope (flat region).
        big_threshold (float): Threshold for detecting significant slope (trend change).
        nodes (np.ndarray): Locations of fuzzy nodes.
        h (float): Width of each fuzzy partition.
        beta_0 (list): List of local intercepts for each node.
        beta_1 (list): List of local slopes for each node.
        ruptures (list): List of detected rupture dictionaries.

    Methods:
        _triangular_basis(x, c, h): Computes the value of the triangular basis function centered at c with width h.
        _build_fuzzy_partition(): Builds the fuzzy partition matrix for the time axis.
        _compute_fuzzy_transform(A): Computes local linear coefficients (intercept and slope) for each node.
        detect_ruptures(): Detects structural breaks based on changes in local slopes.
    """
    def __init__(self, time, values, n_nodes=50, small_threshold=1e-9, big_threshold=5e-8):
        """
        Initializes the detector with time series data and parameters.

        Args:
            time (array-like): Time indices.
            values (array-like): Observed values.
            n_nodes (int): Number of fuzzy nodes (default 50).
            small_threshold (float): Threshold for flat slope (default 1e-9).
            big_threshold (float): Threshold for significant slope (default 5e-8).
        """
        self.time = np.asarray(time)
        self.values = np.asarray(values)
        self.n_nodes = n_nodes
        self.small_threshold = small_threshold
        self.big_threshold = big_threshold
        self.nodes = None
        self.h = None
        self.beta_0 = []
        self.beta_1 = []
        self.ruptures = []

    def _triangular_basis(self, x, c, h):
        """
        Triangular basis function for fuzzy partitioning.

        Args:
            x (np.ndarray): Input array.
            c (float): Center of the triangle.
            h (float): Width of the triangle.

        Returns:
            np.ndarray: Basis function values.
        """
        return np.maximum(1 - np.abs(x - c) / h, 0)

    def _build_fuzzy_partition(self):
        """
        Builds the fuzzy partition matrix for the time axis.

        Returns:
            np.ndarray: Partition matrix (n_nodes x len(time)).
        """
        a, b = self.time.min(), self.time.max()
        self.nodes = np.linspace(a, b, self.n_nodes)
        self.h = (b - a) / (self.n_nodes - 1)
        A = np.array([self._triangular_basis(self.time, c, self.h) for c in self.nodes])
        A = A / A.sum(axis=0)
        return A

    def _compute_fuzzy_transform(self, A):
        """
        Computes local linear coefficients (intercept and slope) for each node.

        Args:
            A (np.ndarray): Fuzzy partition matrix.
        """
        self.beta_0 = []
        self.beta_1 = []
        for k in range(self.n_nodes):
            Ak = A[k]
            ck = self.nodes[k]
            b0 = np.sum(self.values * Ak) / np.sum(Ak)
            b1 = 6 * np.sum(self.values * (self.time - ck) * Ak) / (self.h ** 3 * np.sum(Ak))
            self.beta_0.append(b0)
            self.beta_1.append(b1)

    def detect_ruptures(self):
        """
        Detects structural breaks based on changes in local slopes.

        Returns:
            list or int: List of rupture dictionaries if found, otherwise 0.
        """
        A = self._build_fuzzy_partition()
        self._compute_fuzzy_transform(A)
        beta_1 = np.array(self.beta_1)

        candidates = []
        for k in range(self.n_nodes - 1):
            if abs(beta_1[k]) < self.small_threshold and abs(beta_1[k + 1]) > self.big_threshold:
                candidates.append((k, k + 1))
            elif abs(beta_1[k + 1]) < self.small_threshold and abs(beta_1[k]) > self.big_threshold:
                candidates.append((k + 1, k))

        for k1, k2 in candidates:
            t_start = int(self.nodes[k1] - self.h)
            t_end = int(self.nodes[k2] + self.h)
            t_start = max(t_start, 0)
            t_end = min(t_end, len(self.time) - 1)
            self.ruptures.append({
                "start_time": t_start,
                "end_time": t_end,
                "node_k": int(self.nodes[k1]),
                "node_k+1": int(self.nodes[k2]),
                "beta_1_k": beta_1[k1],
                "beta_1_k+1": beta_1[k2]
            })

        return self.ruptures if self.ruptures else 0

class FastBayesianMDLChangepointDetector:
    """
    FastBayesianMDLChangepointDetector implements a fast, approximate Bayesian Minimum Description Length (BMDL)
    approach for detecting changepoints (structural breaks) in a univariate time series.

    The algorithm uses recursive binary segmentation and a simplified MDL cost function to efficiently
    identify breakpoints where the underlying linear trend changes.

    Attributes:
        time (np.ndarray): Array of time indices.
        values (np.ndarray): Array of observed values.
        n (int): Length of the time series.
        max_breaks (int): Maximum number of breakpoints to detect.
        min_size (int): Minimum segment size allowed between breakpoints.
        breaks (list): List of detected breakpoint indices.

    Methods:
        _fit_segment(x, y):
            Fits a linear regression to the segment (x, y) and returns the sum of squared errors (SSE).
        _mdl_cost(sse, k):
            Computes a simplified MDL cost for a segment with sum of squared errors 'sse' and 'k' breakpoints.
        _binary_segmentation(start, end, depth=0):
            Recursively applies binary segmentation to find breakpoints that minimize the MDL cost.
        detect():
            Runs the changepoint detection algorithm and returns the sorted list of breakpoints (or 0 if none found).
        evaluate_breakpoint(index):
            Evaluates the MDL gain and probability for a specific breakpoint index.
    """
    def __init__(self, time, values, max_breaks=3, min_size=30):
        """
        Initializes the detector with time series data and parameters.

        Args:
            time (array-like): Time indices.
            values (array-like): Observed values.
            max_breaks (int): Maximum number of breakpoints to detect (default 3).
            min_size (int): Minimum segment size allowed between breakpoints (default 30).
        """
        self.time = np.array(time)
        self.values = np.array(values)
        self.n = len(values)
        self.max_breaks = max_breaks
        self.min_size = min_size
        self.breaks = []

    def _fit_segment(self, x, y):
        """
        Fits a linear regression to the segment (x, y) and returns the sum of squared errors (SSE).

        Args:
            x (np.ndarray): Time indices for the segment.
            y (np.ndarray): Observed values for the segment.

        Returns:
            float: Sum of squared errors for the fitted segment.
        """
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        y_hat = model.predict(x.reshape(-1, 1))
        residuals = y - y_hat
        return np.sum(residuals**2)

    def _mdl_cost(self, sse, k):
        """
        Computes a simplified MDL (Minimum Description Length) cost for a segment.

        Args:
            sse (float): Sum of squared errors for the segment.
            k (int): Number of breakpoints in the model.

        Returns:
            float: MDL cost.
        """
        return k * np.log(self.n) + np.log(sse)

    def _binary_segmentation(self, start, end, depth=0):
        """
        Recursively applies binary segmentation to find breakpoints that minimize the MDL cost.

        Args:
            start (int): Start index of the segment.
            end (int): End index of the segment.
            depth (int): Current recursion depth (number of breaks found so far).
        """
        best_cost_reduction = 0
        best_split = None
        full_x = self.time[start:end]
        full_y = self.values[start:end]
        full_sse = self._fit_segment(full_x, full_y)
        full_cost = self._mdl_cost(full_sse, 0)

        for split in range(start + self.min_size, end - self.min_size):
            left_sse = self._fit_segment(self.time[start:split], self.values[start:split])
            right_sse = self._fit_segment(self.time[split:end], self.values[split:end])
            cost = self._mdl_cost(left_sse + right_sse, 1)
            reduction = full_cost - cost
            if reduction > best_cost_reduction:
                best_cost_reduction = reduction
                best_split = split

        if best_split and depth < self.max_breaks:
            self.breaks.append(best_split)
            self._binary_segmentation(start, best_split, depth + 1)
            self._binary_segmentation(best_split, end, depth + 1)

    def detect(self):
        """
        Runs the changepoint detection algorithm and returns the sorted list of breakpoints.

        Returns:
            list or int: Sorted list of detected breakpoints, or 0 if none found.
        """
        self.breaks = []
        self._binary_segmentation(0, self.n)
        self.breaks.sort()
        return self.breaks if self.breaks else 0

    def evaluate_breakpoint(self, index):
        """
        Evaluates if a specific index is a structural break using the MDL criterion.

        Args:
            index (int): Index of the possible breakpoint.

        Returns:
            dict: {
                'mdl_no_break': float,      # MDL cost without a break
                'mdl_with_break': float,    # MDL cost with a break at 'index'
                'mdl_gain': float,          # Gain in MDL (positive means break is favored)
                'probability': float        # Probability (sigmoid of gain)
            }
        """
        if index < self.min_size or index > self.n - self.min_size:
            raise ValueError("El índice está demasiado cerca del inicio o del final para una evaluación válida.")

        full_sse = self._fit_segment(self.time, self.values)
        mdl_no_break = self._mdl_cost(full_sse, 0)

        left_sse = self._fit_segment(self.time[:index], self.values[:index])
        right_sse = self._fit_segment(self.time[index:], self.values[index:])
        mdl_with_break = self._mdl_cost(left_sse + right_sse, 1)

        mdl_gain = mdl_no_break - mdl_with_break
        probability = float(expit(mdl_gain))  # sigmoid to obtain a probability

        return {
            'mdl_no_break': mdl_no_break,
            'mdl_with_break': mdl_with_break,
            'mdl_gain': mdl_gain,
            'probability': probability
        }

class CUSUMTest:
    """
    Clase para realizar el test CUSUM y CUSUM-sq en una serie temporal.

    Permite calcular la estadística de CUSUM (acumulado de desviaciones respecto a la media)
    o CUSUM-sq (acumulado de desviaciones cuadradas respecto a la media) en un punto de ruptura dado,
    y obtener el estadístico de prueba y su p-valor asociado.

    Métodos:
        - __init__(series): Inicializa la clase con una serie temporal (pandas Series).
        - test_statistic(break_point, squared=False): Calcula el estadístico CUSUM o CUSUM-sq y su p-valor
          en el punto de ruptura especificado.
    """
    def __init__(self, series: pd.Series):
        """
        Inicializa la clase con la serie temporal.

        Args:
            series (pd.Series): Serie temporal a analizar.
        """
        self.series = series.dropna().values

    def test_statistic(self, break_point: int, squared: bool = False):
        """
        Calcula la estadística CUSUM o CUSUM-sq en un punto dado.

        Args:
            break_point (int): Índice temporal donde se evalúa el cambio estructural.
            squared (bool): Si True, usa CUSUM-sq; si False, usa CUSUM.

        Returns:
            tuple: (estadístico de prueba, p-valor)
        """
        if break_point <= 0 or break_point >= len(self.series):
            raise ValueError("El punto de ruptura debe estar dentro del rango de la serie.")

        x = self.series
        T = len(x)
        mu = np.mean(x)
        sigma = np.std(x, ddof=1)

        if squared:
            # CUSUM-sq: acumulado de desviaciones cuadradas respecto a la media
            squared_devs = (x - mu) ** 2
            S = np.cumsum(squared_devs - np.mean(squared_devs))
        else:
            # CUSUM: acumulado de desviaciones respecto a la media
            S = np.cumsum(x - mu)

        S_norm = S / (sigma * np.sqrt(T))

        stat = np.abs(S_norm[break_point])
        p_value = 2 * (1 - norm.cdf(stat))

        return stat, p_value

class BaiPerronTest:
    """
    Clase para realizar la prueba de Bai-Perron para detectar un posible cambio estructural
    (breakpoint) en una serie temporal en un punto específico.

    Métodos:
        - __init__(series): Inicializa la clase con una serie temporal (pandas Series).
        - test_statistic(break_point): Calcula el estadístico F y el p-valor para evaluar
          si existe un cambio estructural en el punto especificado.
    """
    def __init__(self, series: pd.Series):
        """
        Inicializa la clase con la serie temporal.

        Args:
            series (pd.Series): Serie temporal a analizar.
        """
        self.series = series.dropna().values

    def test_statistic(self, break_point: int):
        """
        Realiza la prueba Bai-Perron en un punto dado.

        Args:
            break_point (int): Índice donde se evalúa el posible cambio estructural.

        Returns:
            tuple: (F-statistic, p-valor)
        """
        if break_point <= 0 or break_point >= len(self.series):
            raise ValueError("El punto de ruptura debe estar dentro del rango de la serie.")

        y = self.series
        x = np.arange(len(y))
        x = sm.add_constant(x)

        # Ajuste del modelo completo (sin ruptura)
        model_full = sm.OLS(y, x).fit()

        # Ajuste de los modelos segmentados (antes y después del punto de ruptura)
        x1 = sm.add_constant(np.arange(break_point))
        y1 = y[:break_point]
        model1 = sm.OLS(y1, x1).fit()

        x2 = sm.add_constant(np.arange(break_point, len(y)))
        y2 = y[break_point:]
        model2 = sm.OLS(y2, x2).fit()

        # Suma de residuos al cuadrado para cada modelo
        rss_full = np.sum(model_full.resid ** 2)
        rss_split = np.sum(model1.resid ** 2) + np.sum(model2.resid ** 2)

        k = x.shape[1]
        num = (rss_full - rss_split) / k
        den = rss_split / (len(y) - 2 * k)
        f_stat = num / den

        # Cálculo del p-valor asociado al estadístico F
        p_value = 1 - f.cdf(f_stat, dfn=k, dfd=len(y) - 2 * k)

        return f_stat, p_value