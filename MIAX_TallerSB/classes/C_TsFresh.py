# Librerias de sistema
import os
# Librerias Python
import pandas as pd
import numpy as np
# Libreria TsFresh
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.feature_extraction import EfficientFCParameters
# Librerias sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# Librerias Graficas
import matplotlib.pyplot as plt
import seaborn as sns


# Clase para ingeniería de características usando TSFresh
class FeatureEngineerTSFresh:
    def __init__(self, df, id_col='id', time_col='time', value_col='value',
                 target_col='structural_breakpoint', max_ids=None, n_jobs=-1):
        """
        Inicializa el objeto FeatureEngineerTSFresh.

        :param df: DataFrame de entrada.
        :param id_col: Nombre de la columna de ID.
        :param time_col: Nombre de la columna de tiempo.
        :param value_col: Nombre de la columna de valores.
        :param target_col: Nombre de la columna objetivo.
        :param max_ids: Número máximo de IDs a usar (None = todos).
        :param n_jobs: Número de procesos para paralelización (-1 = todos los disponibles).
        """
        # Guarda una copia del DataFrame original con las columnas relevantes
        self.original_df = df[[id_col, time_col, value_col, target_col]].copy()
        self.id_col = id_col
        self.time_col = time_col
        self.value_col = value_col
        self.target_col = target_col
        self.max_ids = max_ids
        self.n_jobs = n_jobs

        # Inicializa atributos para almacenar resultados intermedios
        self.df_filtered = None
        self.feature_df = None
        self.target_series = None
        self.selected_features = None
        self.relevance_table = None

    def filter_ids(self):
        """
        Filtra el DataFrame para usar solo un subconjunto de IDs si se especifica max_ids.
        """
        unique_ids = self.original_df[self.id_col].unique()
        selected_ids = unique_ids[:self.max_ids] if self.max_ids else unique_ids
        self.df_filtered = self.original_df[self.original_df[self.id_col].isin(selected_ids)].copy()

    def prepare_target_series(self):
        """
        Prepara la serie objetivo (target) para la selección de características.
        """
        y = self.df_filtered[[self.id_col, self.target_col]].drop_duplicates()
        y.set_index(self.id_col, inplace=True)
        self.target_series = y[self.target_col]

    def extract_features(self):
        """
        Extrae características usando TSFresh del DataFrame filtrado.
        """
        df_ft = self.df_filtered.drop(columns=[self.target_col])
        self.feature_df = extract_features(df_ft,
                                           column_id=self.id_col,
                                           column_sort=self.time_col,
                                           impute_function=impute,
                                           n_jobs=self.n_jobs,
                                           )

    def calculate_relevance(self):
        """
        Calcula la relevancia de las características extraídas respecto al objetivo.
        """
        self.relevance_table = calculate_relevance_table(self.feature_df, self.target_series,
                                                         ml_task='classification')

    def select_features(self):
        """
        Selecciona las características más relevantes usando TSFresh.
        """
        self.selected_features = select_features(self.feature_df, self.target_series)

    def plot_feature_importance(self, model=None, top_n=20):
        """
        Grafica la importancia de las características seleccionadas usando un modelo RandomForest.

        :param model: Modelo a usar (por defecto RandomForestClassifier).
        :param top_n: Número de características principales a mostrar.
        """
        
        # Verifica que las características y la serie objetivo no estén vacías
        print(f"selected_features shape: {self.selected_features.shape}")
        print(f"target_series shape: {self.target_series.shape}")
        print(f"selected_features.columns: {self.selected_features.columns.tolist()}")
        print(f"target_series.index[:5]:\n{self.target_series.index[:5]}")
        print(f"selected_features.index[:5]:\n{self.selected_features.index[:5]}")
        
        # Verifica si hay características seleccionadas
        if self.selected_features is None or self.selected_features.empty:
            print("No hay características seleccionadas para mostrar.")
            return
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        

        
        # Asegura que las características y el objetivo tengan el mismo índice
        model.fit(self.selected_features, self.target_series)
        importances = pd.Series(model.feature_importances_, index=self.selected_features.columns)
        importances = importances.sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importances.values[:top_n], y=importances.index[:top_n])
        plt.title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        plt.show()

    def export_to_excel(self, filepath="tsfresh_output.xlsx"):
        """
        Exporta los resultados (todas las características, tabla de relevancia y características seleccionadas) a un archivo Excel.

        :param filepath: Ruta del archivo Excel de salida.
        """
        with pd.ExcelWriter(filepath) as writer:
            self.feature_df.to_excel(writer, sheet_name="All Features")
            self.relevance_table.to_excel(writer, sheet_name="Relevance Table")
            self.selected_features.to_excel(writer, sheet_name="Selected Features")
        print(f"Exportación completada: {os.path.abspath(filepath)}")

    def run_all(self):
        """
        Ejecuta todo el flujo de trabajo: filtra IDs, prepara objetivo, extrae características,
        calcula relevancia y selecciona características.
        """
        print("Filtrando IDs...")
        self.filter_ids()
        print("Preparando serie objetivo...")
        self.prepare_target_series()
        print("Extrayendo características (paralelo)...")
        self.extract_features()
        print("Calculando relevancia...")
        self.calculate_relevance()
        print("Seleccionando características...")
        self.select_features()
        print("Proceso completo.")



# Clase para extracción de características multivariante usando TSFresh
class TSFeatureExtractorMultiVariante:
    """
    Esta clase permite extraer características multivariantes de series temporales
    que han sido divididas en dos periodos (por ejemplo, antes y después de un punto de quiebre estructural).
    Utiliza TSFresh para la extracción automática de características.
    """

    def __init__(self, df, id_col='id', time_col='time', value_col='value', period_col='period', breakpoint_col='structural_breakpoint'):
        """
        Inicializa el extractor con el DataFrame y los nombres de las columnas relevantes.

        :param df: DataFrame de entrada con los datos de series temporales.
        :param id_col: Nombre de la columna que identifica cada serie.
        :param time_col: Nombre de la columna de tiempo.
        :param value_col: Nombre de la columna de valores.
        :param period_col: Nombre de la columna que indica el periodo (antes/después del quiebre).
        :param breakpoint_col: Nombre de la columna que indica el punto de quiebre estructural.
        """
        self.df_original = df.copy()
        self.id_col = id_col
        self.time_col = time_col
        self.value_col = value_col
        self.period_col = period_col
        self.breakpoint_col = breakpoint_col
        self.df_dividido = None
        self.df_alineado = None
        self.features = None
        self.features_imputed = None

    def dividir_por_brecha(self):
        """
        Divide los valores de la serie en dos columnas ('value_1' y 'value_2') según el periodo.
        'value_1' corresponde al periodo 0 y 'value_2' al periodo 1.
        """
        try:
            df = self.df_original.sort_values(by=[self.id_col, self.time_col])
            df['value_1'] = np.where(df[self.period_col] == 0, df[self.value_col], np.nan)
            df['value_2'] = np.where(df[self.period_col] == 1, df[self.value_col], np.nan)
            self.df_dividido = df[[self.id_col, self.time_col, 'value_1', 'value_2', self.breakpoint_col]].copy()
            return self.df_dividido
        except Exception as e:
            print(f"Error en dividir_por_brecha: {e}")
            self.df_dividido = None
            return None


    def reindexar_series(self):
        """
        Reindexa y alinea las series temporales para que 'value_1' y 'value_2' tengan la misma longitud,
        permitiendo la comparación directa entre ambos periodos.
        """
        df_sorted = self.df_dividido.sort_values(by=[self.id_col, self.time_col])
        resultados = []

        for serie_id, grupo in df_sorted.groupby(self.id_col):
            value_1 = grupo['value_1'].dropna().reset_index(drop=True)
            value_2 = grupo['value_2'].dropna().reset_index(drop=True)
            min_len = min(len(value_1), len(value_2))

            df_alineado = pd.DataFrame({
                self.id_col: serie_id,
                self.time_col: range(min_len),
                'value_1': value_1[-min_len:].reset_index(drop=True),
                'value_2': value_2[:min_len].reset_index(drop=True),
                self.breakpoint_col: grupo[self.breakpoint_col].iloc[0]
            })
            resultados.append(df_alineado)

        self.df_alineado = pd.concat(resultados, ignore_index=True)
        return self.df_alineado

    def extraer_caracteristicas(self, max_id=2000):
        """
        Extrae características multivariantes usando TSFresh para los primeros 'max_id' IDs.

        :param max_id: Número máximo de IDs a procesar.
        :return: DataFrame con las características extraídas.
        """
        time_series = self.df_alineado.drop(columns=[self.breakpoint_col])
        time_series = time_series[time_series[self.id_col] < max_id]

        self.features = extract_features(
            time_series,
            column_id=self.id_col,
            column_sort=self.time_col,
            default_fc_parameters=EfficientFCParameters(),
            n_jobs=0
        )
        return self.features

    def imputar_caracteristicas(self):
        """
        Imputa los valores faltantes en las características extraídas usando la función de TSFresh.

        :return: DataFrame con las características imputadas.
        """
        self.features_imputed = impute(self.features)
        return self.features_imputed

    def ejecutar_pipeline(self, max_id=2000):
        """
        Ejecuta todo el pipeline: divide por brecha, reindexa, extrae características e imputa.

        :param max_id: Número máximo de IDs a procesar.
        :return: DataFrame final de características imputadas.
        """
        self.dividir_por_brecha()
        self.reindexar_series()
        self.extraer_caracteristicas(max_id)
        self.imputar_caracteristicas()
        return self.features_imputed



class TSFeatureEvaluator:
    """
    Clase para evaluar características extraídas de series temporales usando modelos de clasificación.
    Permite seleccionar características relevantes, entrenar un clasificador, evaluar el desempeño
    y visualizar resultados como la matriz de confusión y la importancia de las características.
    """

    def __init__(self, features, df_original, id_col='id', target_col='structural_breakpoint'):
        """
        Inicializa el evaluador con las características extraídas y el DataFrame original.

        :param features: DataFrame de características extraídas (índice = id).
        :param df_original: DataFrame original con la columna objetivo.
        :param id_col: Nombre de la columna de ID.
        :param target_col: Nombre de la columna objetivo.
        """
        self.features = features
        self.df_original = df_original
        self.id_col = id_col
        self.target_col = target_col
        self.selected_features = None
        self.target_series = None
        self.classifier = None
        self.y_test = None
        self.y_pred = None
        self.feature_importance = None

    def preparar_target(self):
        """
        Prepara la serie objetivo (target) alineando los IDs con las características extraídas.
        """
        self.target_series = self.df_original.groupby(self.id_col)[self.target_col].first()
        self.target_series = self.target_series[self.target_series.index.isin(self.features.index)]

    def seleccionar_caracteristicas(self):
        """
        Selecciona las características más relevantes usando TSFresh.
        Si no se seleccionan características, utiliza todas las disponibles.
        """
        self.selected_features = select_features(self.features, self.target_series)
        if self.selected_features.empty:
            print("\nNo features were selected. Using all features.")
            self.selected_features = self.features
        else:
            print("\nSelected features:")
            print(self.selected_features.head())

        print(f"\nNumber of features: {self.selected_features.shape[1]}")
        print("\nNames of features (first 10):")
        print(self.selected_features.columns.tolist()[:10])

    def entrenar_y_evaluar(self):
        """
        Entrena un clasificador RandomForest y evalúa su desempeño en un conjunto de prueba.
        Muestra la exactitud y el reporte de clasificación.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.selected_features, self.target_series, test_size=0.2, random_state=42
        )
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        self.classifier = clf
        self.y_test = y_test
        self.y_pred = y_pred

        print("\nClassification Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def visualizar_matriz_confusion(self, ruta_img="../img/Features/confusion_matrix.png"):
        """
        Visualiza y guarda la matriz de confusión del modelo entrenado.

        :param ruta_img: Ruta donde se guardará la imagen de la matriz de confusión.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(ruta_img)
        plt.show()

    def visualizar_importancia(self, ruta_img="../img/Features/feature_importance.png", top_n=20):
        """
        Visualiza y guarda la importancia de las características seleccionadas por el modelo.

        :param ruta_img: Ruta donde se guardará la imagen de importancia de características.
        :param top_n: Número de características principales a mostrar.
        """
        importance_df = pd.DataFrame({
            'feature': self.selected_features.columns,
            'importance': self.classifier.feature_importances_
        }).sort_values(by='importance', ascending=False)

        self.feature_importance = importance_df

        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))

        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
        plt.title(f'Top {top_n} Most Important Features')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.savefig(ruta_img)
        plt.show()

    def ejecutar_pipeline(self):
        """
        Ejecuta todo el flujo de evaluación: prepara el target, selecciona características,
        entrena y evalúa el modelo, y visualiza los resultados.
        """
        self.preparar_target()
        self.seleccionar_caracteristicas()
        self.entrenar_y_evaluar()
        self.visualizar_matriz_confusion()
        self.visualizar_importancia()
