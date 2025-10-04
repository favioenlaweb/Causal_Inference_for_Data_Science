# pip install tsfresh
# pip install antropy

import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import antropy
from scipy.stats import energy_distance, wasserstein_distance, ks_2samp

from tqdm import tqdm



class FeaturesToGenerate:
    """Clase para definir qué características queremos generar y cómo queremos
    hacerlo.
    """
    
    # Retardos ################################################################
    diff_lags:list[int]=[3, 5, 7, 22, 30, 90, 180]
    """Indica si se van a hacer cálculos sobre las diferencias entre los 
    valores y sus valores anteriores con los retardos indicados. Es una lista
    de  valores enteros indicando estos retardos; si se indica None, entonces
    no se realizará ningún tipo de cálculo con retardo. No debe indicarse el
    valor cero o todos los cálculos realizados para el retardo cero serán
    degenerados e inútiles. Por defecto es igual a [3, 5, 7, 22, 30, 90, 180].
    """
    
    
    # Ventanas rolling (básicas) ##############################################
    rolling_windows_sizes:list[int]=[5, 10, 25, 50, 100, 150, 200, 250]
    """Indica los tamaños de las ventanas rolling para los cálculos que sean
    indicados en `rolling_windows_calcs`. Si se indica None, entonces no se
    realizarán estos cálculos. Por defecto es igual a [5, 10, 25, 50, 100, 150,
    200, 250] (250 es el valor máximo para las series más cortas).
    """
    
    rolling_windows_calcs:list[str]=['median', 'kurtosis', 'sum', 'cv']
    """Indica los cálculos a realizar en las ventanas rolling. Por ejemplo,
    pueden ser todos los siguientes: ['mean', 'std', 'median', 'min', 'max',
    'skew', 'kurtosis', 'sum', 'cv']. Por defecto, solamente aplica ['median',
    'kurtosis', 'sum', 'cv'], es decir, todos los anteriores que no se generan
    automáticamente con un describe() de pandas. Si es None, no se realizará
    ningún tipo de cálculo simple en las ventanas rolling.
    """
    
        
    # Modelo de descomposición ################################################
    linear_regression:bool=True
    """Indica si utilizar un modelo de regresión lineal para calcular las
    tendencias de la serie temporal. Si es True, entonces se utilizará este
    modelo para calcular las tendencias. Por defecto, es igual a True.
    """
    
    
    # Ventanas rolling avanzadas ##############################################
    advanced_abs_energy:bool=True
    """Indica si generar características de energía o no.
    """
    
    advanced_abs_sum_changes:bool=True
    """Indica si generar características para la suma del valor absoluto de los
    cambios entre un valor y su valor anterior.
    """
    
    advanced_mean_abs_change:bool=True
    """Indica si generar características para la media del valor absoluto de los
    cambios entre un valor y su anterior.
    """


    # Entropía ################################################################
    entropy_dimension:int=2
    """Dimensión de incrustación para los cálculos de entropía utilizando
    antropy.sample_entropy. Un valor menor que 2 hará que este cálculo no se 
    ejecute. Por defecto, es igual a 2.
    """
    
    
    # Autocorrelación #########################################################
    autocorrelation_max_lag:int = 2
    """Número máximo de lags en el cálculo de la autocorrelación. Se utilizarán
    los valores desde el 0 hasta el indicado automáticamente (y, por tanto, no
    tiene sentido indicar valores individuales). Por defecto, es igual a 2.
    """
    
    
    # Frecuency domain (FFT) ##################################################
    num_fft_features:int=5
    """Número de características a generar en el dominio de la frecuencia
    utilizando la FFT. Si es 0 (o menor), entonces no se generará ninguna
    características en el dominio de la frecuencia. Por defecto, es igual
    a 5.
    """
    
    
    # Generar características para la diferencia entre los valores ############
    include_diff_series:bool=True
    """Indica si se han de generar características para la serie de diferencias
    entre los valores tras el punto de break y los valores antes de este mismo
    punto. Si es True, entonces se generarán estas características. Por
    defecto, es igual a True.
    """
    
    
    # Características comparativas ############################################
    include_comparative_features:bool=True
    """Indica si se han de generar características comparativas entre los
    valores antes y después del punto de break. Si es True, entonces se
    generarán estas características. Por defecto, es igual a True.
    """
    
    
    # Delta de características antes y después del punto de break #############
    include_delta_features:bool=True
    """Indica si se han de generar características que representen la 
    diferencia entre las características antes y después del punto de break.
    """
    
    
    def __init__(self):
        """Constructor por defecto de la clase FeaturesToGenerate.
        """
        pass
    
    
    
class FeaturesToReturn:
    """Clase auxiliar para filtrar los tipos de características que queremos 
    utilizar posteriormente, reduciendo el tamaño del `pd.DataFrame` que se
    retorna el método `transform()` de la clase `FeaturesExtractor`. Nótese
    que solo modifica la salida, todas las características que se calculan se
    habrán indicado en una instancia de la clase `FeaturesToGenerate`.
    """
    
    # Características de series individuales ##################################
    caracteristicas_serie_anterior_al_punto_de_break:bool=False
    """Indica si queremos o no retornar las características calculadas para 
    las series de valores anteriores al punto de break en las distintas 
    ventanas a considerar. Esta serie siempre se tomará de forma creciente
    en el tiempo. Por defecto, es igual a False.
    """
    
    caracteristicas_serie_posterior_al_punto_de_break:bool=False
    """Indica si queremos o no retornar las características calculadas para
    las series de valores posteriores al punto de break en las distintas
    ventanas a considerar. Esta serie siempre se tomará de forma creciente
    en el tiempo. Por defecto, es igual a False.
    """
    
    caracteristicas_serie_posterior_al_punto_de_break_invertida:bool=False
    """Indica si queremos o no retornar las características calculadas para
    las series de valores posteriores al punto de break en las distintas
    ventanas a considerar, pero invertidas en el tiempo (es decir, la serie
    posterior al punto de break se toma de forma decreciente en el tiempo).
    Por defecto, es igual a False.
    """
    
    # Características de series conjuntas #####################################
    caracteristicas_series_conjuntas_tiempo_directo:bool=True
    """Indica si queremos retornar o no las características que se han 
    calculado para los pares de series de valores anteriores y posteriores al
    punto de break (en las distintas ventanas del mismo tamaño), donde ambas
    series se han tomado con la misma dirección temporal. Por defecto, es igual
    a True.
    """
    
    caracteristicas_series_conjuntas_tiempo_invertido:bool=True
    """Indica si queremos retornar o no las características que se han
    calculado para los pares de series de valores anteriores y posteriores al
    punto de break (en las distintas ventanas del mismo tamaño), donde la
    serie anterior se ha tomado con la dirección temporal normal (creciente)
    y la posterior se ha tomado con la dirección temporal invertida
    (decreciente). Por defecto, es igual a True.
    """
    
    # Características series de diferencias ###################################
    caracteristicas_serie_diferencias_tiempo_directo:bool=True
    """Indica si queremos retornar o no las características que se han
    calculado para las series de diferencias entre los valores anteriores y
    posteriores al punto de break (en las distintas ventanas del mismo
    tamaño), donde ambas series se han tomado con la misma dirección
    temporal. Por defecto, es igual a True.
    """
    
    caracteristicas_serie_diferencias_tiempo_invertido:bool=True
    """Indica si queremos retornar o no las características que se han
    calculado para las series de diferencias entre los valores anteriores y
    posteriores al punto de break (en las distintas ventanas del mismo
    tamaño), donde la serie anterior se ha tomado con la dirección temporal
    normal (creciente) y la posterior se ha tomado con la dirección temporal
    invertida (decreciente). Por defecto, es igual a True.
    """
    
    # Características de diferencias entre series #############################
    diferencia_caracteristicas_series_tiempo_directo:bool=True
    """Indica si queremos retornar o no las diferencias entre las 
    características calculadas para las series de valores anteriores y
    posteriores al punto de break (en las distintas ventanas del mismo
    tamaño), donde ambas series se han tomado con la misma dirección
    temporal. Por defecto, es igual a True.
    """
    
    diferencia_caracteristicas_series_tiempo_invertido:bool=True
    """Indica si queremos retornar o no las diferencias entre las
    características calculadas para las series de valores anteriores y
    posteriores al punto de break (en las distintas ventanas del mismo
    tamaño), donde la serie anterior se ha tomado con la dirección temporal
    normal (creciente) y la posterior se ha tomado con la dirección temporal
    invertida (decreciente). Por defecto, es igual a True.
    """
    
    
    def __init__(self):
        """Constructor por defecto de la clase FeaturesToReturn.
        """
        pass
    
    
    def activar_todas(
        self, 
        tiempo_directo:bool=True, 
        tiempo_invertido:bool=False
    ) -> None:
        """Activa todas las características de la clase según el tipo de
        dirección temporal que se indique. Si `tiempo_directo` es True, 
        entonces se activarán todas las características que se toman
        con la dirección temporal normal (creciente), mientras que si
        `tiempo_invertido` es True, entonces se activarán todas las
        características que se toman con la dirección temporal invertida
        (decreciente). Si ambos son True, entonces se activarán todas las
        características de la clase. Si ambos son False, entonces no se
        activará ninguna característica de la clase, salvo la correspondiente
        a las características asociadas a la serie de valores anterior al 
        punto de break, que siempre se activará.
        
        Args:
            tiempo_directo (bool, optional): Indica si se han de activar las
            características que se toman con la dirección temporal normal
            (creciente). Por defecto, es igual a True.
            tiempo_invertido (bool, optional): Indica si se han de activar las
            características que se toman con la dirección temporal invertida
            (decreciente). Por defecto, es igual a False.
        """
        
        self.caracteristicas_serie_anterior_al_punto_de_break = True
        self.caracteristicas_serie_posterior_al_punto_de_break = tiempo_directo
        self.caracteristicas_serie_posterior_al_punto_de_break_invertida = tiempo_invertido
        self.caracteristicas_series_conjuntas_tiempo_directo = tiempo_directo
        self.caracteristicas_series_conjuntas_tiempo_invertido = tiempo_invertido
        self.caracteristicas_serie_diferencias_tiempo_directo = tiempo_directo
        self.caracteristicas_serie_diferencias_tiempo_invertido = tiempo_invertido
        self.diferencia_caracteristicas_series_tiempo_directo = tiempo_directo
        self.diferencia_caracteristicas_series_tiempo_invertido = tiempo_invertido
    
    
    def restablecer_valores_por_defecto(self) -> None:
        """Restablece los valores por defecto de la clase.
        """
        
        self.caracteristicas_serie_anterior_al_punto_de_break = False
        self.caracteristicas_serie_posterior_al_punto_de_break = False
        self.caracteristicas_serie_posterior_al_punto_de_break_invertida = False
        self.caracteristicas_series_conjuntas_tiempo_directo = True
        self.caracteristicas_series_conjuntas_tiempo_invertido = True
        self.caracteristicas_serie_diferencias_tiempo_directo = True
        self.caracteristicas_serie_diferencias_tiempo_invertido = True
        self.diferencia_caracteristicas_series_tiempo_directo = True
        self.diferencia_caracteristicas_series_tiempo_invertido = True
                
        
    def __str__(self) -> str:
        """Representación en cadena de la clase.
        
        Returns:
            str: Representación en cadena de la clase.
        """
        
        return (
            f"FeaturesToReturn(\n"
            f"    caracteristicas_serie_anterior_al_punto_de_break={self.caracteristicas_serie_anterior_al_punto_de_break},\n"
            f"    caracteristicas_serie_posterior_al_punto_de_break={self.caracteristicas_serie_posterior_al_punto_de_break},\n"
            f"    caracteristicas_serie_posterior_al_punto_de_break_invertida={self.caracteristicas_serie_posterior_al_punto_de_break_invertida},\n"
            f"    caracteristicas_series_conjuntas_tiempo_directo={self.caracteristicas_series_conjuntas_tiempo_directo},\n"
            f"    caracteristicas_series_conjuntas_tiempo_invertido={self.caracteristicas_series_conjuntas_tiempo_invertido},\n"
            f"    caracteristicas_serie_diferencias_tiempo_directo={self.caracteristicas_serie_diferencias_tiempo_directo},\n"
            f"    caracteristicas_serie_diferencias_tiempo_invertido={self.caracteristicas_serie_diferencias_tiempo_invertido},\n"
            f"    diferencia_caracteristicas_series_tiempo_directo={self.diferencia_caracteristicas_series_tiempo_directo},\n"
            f"    diferencia_caracteristicas_series_tiempo_invertido={self.diferencia_caracteristicas_series_tiempo_invertido}\n"
            f")"
        )



class FeaturesExtractor:
    """Clase encargada de la generación de características sobre un objeto de
    tipo pd.DataFrame aplanado.
    """
    
    __features:FeaturesToGenerate
    """Objeto donde están definidas las características a generar sobre los
    pd.DataFrame.
    """
    
    def __init__(
        self,
        features:FeaturesToGenerate=None
    ):
        """Constructor de la clase.

        Args:
            features (FeaturesToGenerate, optional): Objeto donde se han 
            definido todas las características a generar y qué parámetros
            han de utilizarse en esta generación. Por defecto, es None, en
            cuyo caso se utiliza un objeto FeaturesToGenerate con todos sus
            valores por defecto.
        """
        
        self.__features = (features if features else FeaturesToGenerate())
        pass
    
    
    def transform(
        self,
        df:pd.DataFrame,
        index_col_name:str='id', 
        time_col_name:str='time', 
        values_col_name:str='value',
        period_col_name:str='period',
        normalize:bool=False,
        features_to_return:FeaturesToReturn=None
    ) -> pd.DataFrame:
        """Genera todas las características configuradas sobre el pd.DataFrame
        indicado. La salida es un DataFrame cuyo índice es el índice que se
        corresponde con los datos indicados y su valor es otro diccionario cuya
        clave es 0 o 1 (0 = características generadas antes del punto de break;
        1 = características generadas después del punto de break) y sus valores
        son objetos pd.DataFrame con las características generadas. Nótese que
        estos últimos pd.DataFrame tendrán siempre la columna de tiempo y, de
        forma opcional, la de valores.        

        Args:
            df (pd.DataFrame): DataFrame de pandas con los datos de entrada. No
            es necesario hacer una copia del mismo previamente, ya que se copia
            durante la ejecución de esta función.
            index_col_name (str, optional): Nombre de la columna de `df` en la
            cual están todos los índices que dividen el DataFrame en bloques.
            Por defecto, es igual a 'id'.
            time_col_name (str, optional): Nombre de la columna de `df` donde 
            están las marcas de tiempo. Esta columna siempre se copiará en los
            pd.DataFrame de salida. Por defecto, es igual a 'time'.
            values_col_name (str, optional): Nombre de la columna de `df` en la
            cual están los valores a tratar. Por defecto, es igual a 'value'.
            period_col_name (str, optional): Nombre de la columna de `df` donde
            está el indicador de si el dato es previo al punto de break o si es
            posterior. Por defecto, es igual a 'period'.
            include_values_out (bool, optional): Indica si se ha de incluir la
            columna de valores (`values_col_name`) en los pd.DataFrame que son
            generados por la transformación (obviamente, cada pd.DataFrame que
            se genere solamente tendrá una sección de esta columna). Su valor
            por defecto es igual a False.
            normalize (bool, optional): Indica si se han de normalizar los
            valores de la columna `values_col_name` antes de generar las
            características. Si es True, entonces se normalizarán los valores
            de la columna `values_col_name` para cada índice de forma 
            individual e independiente (una única vez para toda la serie, no se
            normalizará en cada ventana).
            features_to_return (FeaturesToReturn, optional): Objeto donde se
            indican las características que se han de retornar en el DataFrame
            de salida. Si es None, entonces se retornarán las definidas en los
            valores por defecto de la clase `FeaturesToReturn`. Por defecto, es
            igual a None.

        Returns:
            pd.DataFrame: DataFrame de salida con las características generadas
            (el índice es el índice de los datos originales y las columnas son
            los valores de las características generadas). Hay varios tipos de
            características y todas pueden ser controladas a través del objeto
            de parámetros que haya suministrado al constructor de la clase.
        """
        
        df_copia = df.copy()
        df_copia.sort_values(by=[index_col_name, time_col_name], inplace=True)
        
        dict_elementos = { }
        
        # Dado que piden que cada index se trate de forma independiente en la
        # ejecución de la inferencia y, por simplicidad, vamos a hacer un for
        # para cada índice. No va a ser lo más eficiente, pero quedará de una
        # forma más fácil de seguir.
        for index, df_indice in tqdm(df_copia.groupby(index_col_name)):
            
            if normalize:
                serie = df[values_col_name]
                std = serie.std()
                if std >= 1E-8:
                    media = serie.mean()
                    df.loc[:, values_col_name] = (serie - media) / std
                else:
                    df.loc[:, values_col_name] = 0
            
            dict_ventanas = self.__generar_ventanas_rolling(
                df_indice,
                self.__features.rolling_windows_sizes,
                values_col_name=values_col_name,
                period_col_name=period_col_name,
                include_diff_series=self.__features.include_diff_series
            )
            
            dict_series = dict_ventanas['series']
            dict_pares = dict_ventanas['pares']
            
            caracteristicas = { }
            
            for key, serie in dict_series.items():
                caracteristicas.update(self.__generar_caracteristicas_grupo(
                    serie,
                    self.__features,
                    key,
                    '',
                    False
                ))
                
                if self.__features.diff_lags:
                    for lag in self.__features.diff_lags:
                        if lag > len(serie) - 5:
                            # Si el lag es demasiado grande, no tiene sentido
                            # calcularlo.
                            continue
                        
                        valores_lag = serie.shift(lag)
                        valores_diff = (serie - valores_lag).dropna()
                                                
                        caract_diff_lag = self.__generar_caracteristicas_grupo(
                            valores_diff,
                            self.__features,
                            key,
                            f'_serielag{lag:03}',
                            True
                        )
                        
                        caracteristicas.update(caract_diff_lag)
                        
                df_acum = serie.cumsum().iloc[1:].reset_index(drop=True)
                caract_acum = self.__generar_caracteristicas_grupo(
                    df_acum,
                    self.__features,
                    key,
                    '_serieacum',
                    False
                )
                
                caracteristicas.update(caract_acum)
            
            if (self.__features.include_comparative_features):
                for key, (serie_p0, serie_p1) in dict_pares.items():
                    # Generamos las características comparativas:
                    caracteristicas.update(
                        self.__generar_caracteristicas_comparativas(
                            serie_p0,
                            serie_p1,
                            prefijo=key
                        )
                    )
            
            dict_elementos[index] = caracteristicas
        
        
        res = pd.DataFrame.from_dict(
            dict_elementos, 
            orient='index'
        )

        # Vamos a generar diferencias entre los valores previos y posteriores,
        # de forma que tengamos características adicionales que comparen los
        # valores antes y después del punto de break (los árboles de decisión 
        # toman decisiones de la forma "característica frente a valor", e.d.,
        # no miran "característica frente a característica", por lo que añado
        # estas diferencias para que puedan ser incluidas en los árboles; hay
        # algunas que serán redundantes, pues los rolling_diff ya las calculan,
        # pero así cubrimos más casos):
        # Este código no es óptimo, pero es fácil de seguir:
        if (self.__features.include_delta_features):
            columnas_delta = { }
            for clave in res.columns:
                if 'rolling_p0_w' in clave:
                    previo = res[clave]
                    
                    # Comparativa directa (p0[t↑] frente a p1[t↑])    
                    alt = clave.replace(
                        'rolling_p0_w', 
                        'rolling_p1_w'
                    )
                    nueva_clave = clave.replace(
                        'rolling_p0_w', 
                        'rolling_delta_dir_w'
                    )
                    if alt in res.columns:
                        posterior = res[alt]
                        columnas_delta[nueva_clave] = posterior - previo
                    
                    # Comparativa inversa (p0[t↑] frente a p1[t↓])
                    alt = clave.replace(
                        'rolling_p0_w', 
                        'rolling_p1_inv_w'
                    )
                    nueva_clave = clave.replace(
                        'rolling_p0_w', 
                        'rolling_delta_inv_w'
                    )
                    if alt in res.columns:
                        posterior = res[alt]
                        columnas_delta[nueva_clave] = posterior - previo
            
            if columnas_delta:
                df_delta = pd.DataFrame(
                    columnas_delta, 
                    index=res.index
                )
                res = pd.concat([res, df_delta], axis=1)
        
        res = self.__filtrar_resultados(res, features_to_return)
        
        return res
    
    
    def __generar_ventanas_rolling(
        self,
        df_index:pd.DataFrame,
        windows_sizes:list[int],
        values_col_name:str='value',
        period_col_name:str='period',
        include_diff_series:bool=False
    ) -> dict[str, object]:
        """Genera las ventanas rolling que se utilizarán en los cálculos de
        características.
        
        Args:
            df_index (pd.DataFrame): DataFrame con los datos de entrada para un
            índice concreto.
            windows_sizes (list[int]): Tamaños de las ventanas rolling a 
            utilizar. Si es None o una colección vacía, entonces se generará
            solamente una ventana de tamaño 100.
            values_col_name (str): Nombre de la columna de `df` donde están los
            valores a tratar. Por defecto, es igual a 'value'.
            period_col_name (str): Nombre de la columna de `df` donde está el
            indicador de si el dato es previo al punto de break o si, por el
            contrario, es posterior. Por defecto, es igual a 'period'.
            include_diff_series (bool): Indica si se han de generar las series
            de diferencias entre los valores previos y posteriores a la ventana
            rolling. Si es True, entonces se generarán estas series de
            diferencias. Por defecto, es igual a False.
            
        Returns:
            dict[str, object]: Diccionario con las ventanas rolling generadas,
            organizadas en otros dos diccionarios. Llamando `d` a la salida,
            entonces `d['series']` es un diccionario cuya clave es el nombre de
            la ventana y el valor es un objeto pd.DataFrame con los datos de 
            esa ventana, mientras que `d['pares']` es otro diccionario cuya 
            clave es un identificador para cada par y su valor es una tupla con
            las dos ventanas de igual tamaño generadas a partir de la serie.
        """
    
        if (not windows_sizes or (windows_sizes is None)):
            windows_sizes = [100]
            
        series = { }
        pares = { }
        
        df_previos = df_index[df_index[period_col_name] == 0]
        df_posteriores = df_index[df_index[period_col_name] == 1]
        
        len_previos = len(df_previos)
        len_posteriores = len(df_posteriores)
        
        for window_size in windows_sizes:
            if window_size <= 0:
                print(
                    f"Advertencia: Tamaño de ventana rolling no válido: " +
                    f"{window_size}. Omitiendo."
                )
                continue
                        
            # Calculamos el tamaño "seguro" para las ventanas, de forma que 
            # tengan la misma longitud antes y después del punto de break.
            safe_size = min(len_previos, len_posteriores, window_size)
            
            # Generamos la ventana rolling para los datos previos (que se toma
            # desde el final de la serie):
            rolling_previos = df_previos.tail(safe_size)
            serie_previos = rolling_previos[values_col_name]
            
            # Generamos la ventana rolling para los datos posteriores (que se
            # toma desde el principio de la serie):
            rolling_posteriores = df_posteriores.head(safe_size)
            serie_posteriores = rolling_posteriores[values_col_name]
            serie_invertida = serie_posteriores[::-1].reset_index(drop=True)
            
            # Añadimos los resultados al diccionario de salida:
            series[f'rolling_p0_w{window_size}'] = serie_previos
            series[f'rolling_p1_w{window_size}'] = serie_posteriores
            series[f'rolling_p1_inv_w{window_size}'] = serie_invertida
            pares[f'par_dir_w{window_size}'] = (serie_previos, serie_posteriores)
            pares[f'par_inv_w{window_size}'] = (serie_previos, serie_invertida)
            
            if include_diff_series:
                # Calculamos la diferencia entre los valores previos y posteriores
                # para tener un conjunto de características adicionales:
                series[f'rolling_diff_dir_w{window_size}'] = (
                    serie_posteriores.reset_index(drop=True)
                    - serie_previos.reset_index(drop=True)
                )
                
                series[f'rolling_diff_inv_w{window_size}'] = (
                    serie_posteriores.reset_index(drop=True)
                    - serie_previos.reset_index(drop=True)
                )
            
        res = {
            'series': series,
            'pares': pares
        }
            
        return res
    
    
    def __filtrar_resultados(
        self,
        df:pd.DataFrame,
        features_to_return:FeaturesToReturn=None
    ) -> pd.DataFrame:
        """Filtra el DataFrame de salida para que solamente contenga las
        características que se han indicado en el objeto `features_to_return`.
        
        Args:
            df (pd.DataFrame): DataFrame de salida con las características
            generadas.
            features_to_return (FeaturesToReturn, optional): Objeto donde se
            indican las características que se han de retornar en el DataFrame
            de salida. Si es None, entonces se retornarán las definidas en los
            valores por defecto de la clase `FeaturesToReturn`. Por defecto, es
            igual a None.
        """
        
        ftr = (
            features_to_return 
            if features_to_return
            else FeaturesToReturn()
        )
        
        prefijos_columnas = []
        
        if (ftr.caracteristicas_serie_anterior_al_punto_de_break):
            prefijos_columnas.append('rolling_p0_w')
        if (ftr.caracteristicas_serie_posterior_al_punto_de_break):
            prefijos_columnas.append('rolling_p1_w')
        if (ftr.caracteristicas_serie_posterior_al_punto_de_break_invertida):
            prefijos_columnas.append('rolling_p1_inv_w')
        if (ftr.caracteristicas_series_conjuntas_tiempo_directo):
            prefijos_columnas.append('par_dir_w')
        if (ftr.caracteristicas_series_conjuntas_tiempo_invertido):
            prefijos_columnas.append('par_inv_w')   
        if (ftr.caracteristicas_serie_diferencias_tiempo_directo):
            prefijos_columnas.append('rolling_diff_dir_w')
        if (ftr.caracteristicas_serie_diferencias_tiempo_invertido):
            prefijos_columnas.append('rolling_diff_inv_w')
        if (ftr.diferencia_caracteristicas_series_tiempo_directo):
            prefijos_columnas.append('rolling_delta_dir_w')
        if (ftr.diferencia_caracteristicas_series_tiempo_invertido):
            prefijos_columnas.append('rolling_delta_inv_w')
            
        res = None
        if len(prefijos_columnas) > 0:
            columnas = [
                col
                for col in df.columns
                if any(
                    col.startswith(prefijo) 
                    for prefijo in prefijos_columnas
                )
            ]
            res = df[columnas]
        else:
            res = df
        
        return res

        
    def __generar_caracteristicas_grupo(
        self,
        serie:pd.Series,
        caracteristicas:FeaturesToGenerate,
        prefijo_nombres:str,
        sufijo_nombres:str,
        indicador_calculo_lag:bool
    ) -> dict[str, float]:
        """Genera las características configuradas para un grupo.

        Args:
            serie (pd.Series): Datos del grupo.
            caracteristicas (FeaturesToGenerate): Definición de las 
            características a generar, junto a su configuración.
            prefijo_nombres (str): Prefijo para el nombre de las columnas
            que se generen.
            sufijo_nombres (str): Sufijo para el nombre de las columnas
            que se generen.
            indicador_calculo_lag (bool): Indica si estamos haciendo 
            cálculos con lag, de forma que se excluyan algunos de los
            estadísticos que dejan de tener sentido en este momento.

        Returns:
            dict[str, float]: Diccionario con las características que 
            han sido generadas, en donde la clave es el nombre de la
            característica y el valor su valor numérico calculado.
        """
        
        res = self.__generar_caracteristicas_descriptivas(
            serie,
            prefijo_nombres,
            sufijo_nombres
        )
        
        if (
            caracteristicas.rolling_windows_calcs 
        ):
            res.update(self.__generar_caracteristicas_basicas(
                serie,
                caracteristicas.rolling_windows_calcs,
                prefijo_nombres,
                sufijo_nombres
            ))
                        
        if caracteristicas.linear_regression:
            res.update(self.__generar_caracteristicas_regresion_lineal(
                serie,
                prefix=prefijo_nombres,
                suffix=sufijo_nombres
            ))
            
        
        if (
            caracteristicas.rolling_windows_sizes and (
                caracteristicas.advanced_abs_energy or
                caracteristicas.advanced_abs_sum_changes or
                caracteristicas.advanced_mean_abs_change
            )
        ):
            res.update(self.__generar_caracteristicas_avanzadas(
                serie,
                caracteristicas.advanced_abs_energy,
                caracteristicas.advanced_abs_sum_changes,
                caracteristicas.advanced_mean_abs_change,
                prefix=prefijo_nombres,
                suffix=sufijo_nombres
            ))
            
        if caracteristicas.entropy_dimension >= 2:
            res.update(self.__generar_caracteristicas_entropia(
                serie,
                caracteristicas.entropy_dimension,
                prefix=prefijo_nombres,
                suffix=sufijo_nombres
            ))
        
        if (
            (indicador_calculo_lag == False) and
            caracteristicas.autocorrelation_max_lag > 0
        ):
            res.update(self.__generar_caracteristicas_autocorrelacion(
                serie,
                caracteristicas.autocorrelation_max_lag,
                prefix=prefijo_nombres,
                suffix=sufijo_nombres
            ))
            
        if (caracteristicas.num_fft_features > 0):
            res.update(self.__generar_caracteristicas_frecuencia(
                serie,
                caracteristicas.num_fft_features,
                prefix=prefijo_nombres,
                suffix=sufijo_nombres
            ))
        
        return res
    
    
    def __generar_caracteristicas_basicas(
        self,
        serie:pd.Series,
        windows_calcs:list[str],
        prefijo_nombres:str,
        sufijo_nombres:str
    ) -> dict[str, float]:
        """Genera características básicas.

        Args:
            serie (pd.Series): Serie sobre la cual se generarán las 
            características.
            windows_sizes (list[int]): Tamaños de ventana rolling a utilizar.
            windows_calcs (list[str]): Nombres de los cálculos a generar. Han 
            de ser 'skew', 'kurtosis', 'cv' o nombres de funciones reales que
            se puedan ejecutar sobre un objeto pd.Series.
            prefijo_nombres (str): Prefijo para asignar al nombre de todas las
            características que se generen.
            sufijo_nombres (str): Sufijo para el nombre de las características
            que se generen.

        Returns:
            dict[str, float]: Diccionario con las características generadas.
            La clave es el nombre de la característica y el valor es el valor
            numérico de la característica calculado.
        """
        
        res = { }
        
        for calc in windows_calcs:
            feature_name = f'{prefijo_nombres}_{calc}{sufijo_nombres}'
            if calc == 'skew':
                res[feature_name] = skew(serie)
            elif calc == 'kurtosis':
                res[feature_name] = kurtosis(serie)
            elif calc == 'cv': # Coeficiente de variación
                rolling_mean = serie.mean()
                rolling_std = serie.std()
                res[feature_name] = np.where(
                    rolling_mean != 0,
                    (rolling_std / rolling_mean),
                    np.nan
                )
            elif hasattr(serie, calc):
                res[feature_name] = getattr(serie, calc)()
            else:
                print(
                    f"Advertencia: La estadística '{calc}' no es directament" +
                    "e soportada por un objeto pd.Series. Omitiendo."
                )
    
        return res
    
    
    def __generar_caracteristicas_descriptivas(
        self,
        serie:pd.Series,
        prefijo_nombres:str,
        sufijo_nombres:str
    ) -> dict[str, float]:
        """Genera características descriptivas de la serie.
        Args:
            serie (pd.Series): Serie sobre la cual se generarán las
            características descriptivas.
            prefijo_nombres (str): Prefijo para asignar al nombre de todas las
            características que se generen.
            sufijo_nombres (str): Sufijo para el nombre de las características
            que se generen.
            
        Returns:
            dict[str, float]: Diccionario con las características descriptivas
            generadas. La clave es el nombre de la característica y el valor es
            el valor numérico de la característica calculado.
        """
        
        prefijo_nombres = f'{prefijo_nombres}_descriptivo'
        
        descripcion = serie.describe()
        res = {
            f'{prefijo_nombres}_mean{sufijo_nombres}': descripcion['mean'],
            f'{prefijo_nombres}_std{sufijo_nombres}': descripcion['std'],
            f'{prefijo_nombres}_min{sufijo_nombres}': descripcion['min'],
            f'{prefijo_nombres}_25%{sufijo_nombres}': descripcion['25%'],
            f'{prefijo_nombres}_50%{sufijo_nombres}': descripcion['50%'],
            f'{prefijo_nombres}_75%{sufijo_nombres}': descripcion['75%'],
            f'{prefijo_nombres}_max{sufijo_nombres}': descripcion['max'],
            f'{prefijo_nombres}_var{sufijo_nombres}': descripcion['std'] ** 2,
        }
        
        return res
    
    
    def __generar_caracteristicas_regresion_lineal(
        self,
        serie:pd.Series,
        prefix:str,
        suffix:str
    ) -> dict[str, float]:
        """Genera características de regresión lineal.

        Args:
            serie (pd.Series): Serie sobre la cual se generarán las 
            características.
            prefix (str): Prefijo para el nombre de las características que
            se generen.
            suffix (str): Sufijo para el nombre de las características que
            se generen.

        Returns:
            dict[str, float]: Diccionario con las características generadas.
            La clave es el nombre de la característica y el valor es el valor
            numérico de la característica calculado.
        """
        
        x = np.arange(len(serie)).reshape(-1, 1)
        y = serie.values.reshape(-1, 1)
        
        # Regresión lineal
        regr = linear_model.LinearRegression()
        regr.fit(x, y)

        coeficente = regr.coef_[0][0]       
        termino_indpt = regr.intercept_[0]
        mse = mean_squared_error(x, y)
        varianza = r2_score(x, y)
        
        res = {
            f'{prefix}_rl_coeficiente{suffix}': coeficente,
            f'{prefix}_rl_termino_independiente{suffix}': termino_indpt,
            f'{prefix}_rl_mse{suffix}': mse,
            f'{prefix}_rl_varianza{suffix}': varianza
        }
        
        return res


    def __generar_caracteristicas_avanzadas(
        self,
        serie:pd.Series,
        calc_abs_energy:bool,
        calc_abs_sum_changes:bool,
        calc_mean_abs_change:bool,
        prefix:str,
        suffix:str
    ) -> dict[str, float]:
        """Calculas estadísticos "avanzados".

        Args:
            serie (pd.Series): Serie sobre la cual se generarán todas las
            características.
            calc_abs_energy (bool): Indica si generar datos de energía.
            calc_abs_sum_changes (bool): Indica si generar datos de suma del
            valor absoluto de los cambios.
            calc_mean_abs_change (bool): Indica si generar datos de la media
            del valor absoluto de los cambios.
            prefix (str): Prefijo para el nombre de las columnas que se creen.
            suffix (str): Sufijo para el nombre de las columnas que se creen.

        Returns:
            dict[str, float]: Diccionario con las características generadas.
            La clave es el nombre de la característica y el valor es el valor
            numérico de la característica calculado.
        """
        
        res = { }
                
        if calc_abs_energy:
            caract_name = f'{prefix}_abs_energy{suffix}'
            res[caract_name] = np.nansum(np.square(serie.values))
        
        if calc_abs_sum_changes:
            caract_name = f'{prefix}_abs_sum_changes{suffix}'
            res[caract_name] = np.nansum(np.abs(np.diff(serie.values)))
        
        if calc_mean_abs_change:
            caract_name = f'{prefix}_mean_abs_change{suffix}'
            res[caract_name] = (
                np.nanmean(np.abs(np.diff(serie.values)))
                if np.sum(np.isnan(np.diff(serie.values)) == False) >= 1
                else np.nan
            )
            
        return res


    def __generar_caracteristicas_entropia(
        self,
        serie:pd.Series,
        entropy_dimension:int,
        prefix:str,
        suffix:str
    ) -> dict[str, float]:
        """Cálculos para estadísticas de entropía.

        Args:
            serie (pd.Series): Serie sobre la cual se va a calcular la 
            entropía.
            entropy_dimension (int): Dimensión de incrustación.
            prefix (str): Prefijo para el nombre de las columnas que se creen.
            suffix (str): Sufijo para el nombre de las columnas que se creen.

        Returns:
            pd.DataFrame: DataFrame de características generado.
        """
        
        longitud = len(serie)
        num_nas = pd.isna(serie).sum()
        num_datos = (longitud - num_nas)
        
        ans = 0
        
        if (num_datos < (entropy_dimension + 1)):
            # antropy necesita al menos dimension + 1 puntos
            ans = np.nan
        else:
            std_datos = np.std(serie.values)
            if (abs(std_datos) < 1E-16):
                # Si std es 0, la entropía no está bien definida o es 0.
                ans = 0.0
            else:
                try:
                    ans = antropy.sample_entropy(
                        serie.values,
                        order=entropy_dimension,
                        metric='chebyshev'
                    )
                except Exception:
                    # Puede fallar por varias razones, como, por ejemplo,
                    # si la desviación típica es demasiado pequeña.
                    ans = np.nan
        
        res = {
            f'{prefix}_entropy{suffix}': ans
        }
        
        return res


    def __generar_caracteristicas_autocorrelacion(
        self,
        serie:pd.Series,
        max_lag:int,
        prefix:str,
        suffix:str
    ) -> dict[str, float]:
        """Genera características de autocorrelación.

        Args:
            serie (pd.Series): Serie sobre la cual se van a realizar los 
            cálculos de características.
            max_lag (int): Número máximo de lags en el cálculo de la 
            autocorrelación. Se utilizarán los valores desde el 0 hasta el 
            indicado automáticamente (y, por tanto, no tiene sentido indicar
            valores individuales).
            prefix (str): Prefijo para el nombre de las columnas que se creen.
            suffix (str): Sufijo para el nombre de las columnas que se creen.

        Returns:
            dict[str, float]: Diccionario con las características de
            autocorrelación generadas. La clave es el nombre de la
            característica y el valor es el valor numérico de la 
            característica calculado.
        """
        
        def calculate_acf_at_lags(
            x:pd.Series,
            lag_maximo:int
        ):            
            ans = None
            
            if pd.isna(x).all() or len(x) <= lag_maximo: 
                # No tenemos suficientes puntos para el cálculo
                ans = [np.nan] * (lag_maximo + 1)
            
            try:
                acf_values = acf(
                    x, 
                    nlags=lag_maximo, 
                    fft=True,
                    missing='drop'
                ) 
                ans = [
                    acf_values[lag] 
                    if lag < len(acf_values) 
                    else np.nan 
                    for lag in range(lag_maximo + 1)
                ]
            except Exception:
                ans = [np.nan] * (lag_maximo + 1)
                
            return ans
        
        
        autocorrelacion = calculate_acf_at_lags(serie, max_lag)
        res = { 
            f'{prefix}_autocorrelation_lag{lag}{suffix}': autocorrelacion[lag]
            for lag in range(max_lag + 1)
        }
        
        return res
    
    
    def __generar_caracteristicas_frecuencia(
        self,
        serie:pd.Series,
        num_features:int,
        prefix:str,
        suffix:str
    ) -> dict[str, float]:
        """Genera características en el dominio de la frecuencia utilizando la
        FFT.
        
        Args:
            serie (pd.Series): Serie sobre la cual se van a realizar los
            cálculos de características.
            num_features (int): Número de características a generar en el
            dominio de la frecuencia utilizando la FFT. Siempre será mayor
            que cero.
            prefix (str): Prefijo para el nombre de las columnas que se creen.
            suffix (str): Sufijo para el nombre de las columnas que se creen.
            
        Returns:
            dict[str, float]: Diccionario con las características de frecuencia
            generadas. La clave es el nombre de la característica y el valor es
            el valor numérico de la característica calculado.
        """
        
        def calculate_fft_features(
            x:np.ndarray,
            num_caract_fft:int
        ) -> list[float]:
            ans = [np.nan] * num_caract_fft
            
            # Trabajaremos con valores de numpy y quitamos los NaN
            valores = x.dropna().values
            
            # Si no hay suficientes datos o todos son NaN, no podemos calcular
            # la FFT.
            
            if len(valores) >= num_caract_fft:
                try:
                    # Calculamos la FFT de los valores
                    fft_coeffs = fft(valores)
                
                    # Solo la primera mitad de los coeficientes es relevante
                    num_coeffs_utiles = len(fft_coeffs) // 2
                    hasta = min(num_coeffs_utiles, num_caract_fft)
                    
                    if num_coeffs_utiles > 0:
                        fft_magnitudes = np.abs(fft_coeffs[:num_coeffs_utiles])
                        ans[:hasta] = fft_magnitudes[:hasta].tolist()
                        
                except Exception:
                    pass
                
            return ans
        
        caracteristicas_fft = calculate_fft_features(
            x=serie,
            num_caract_fft=num_features
        )
        
        res = {
            f'{prefix}_fft_n{i}{suffix}': caracteristicas_fft[i]
            for i in range(num_features)
        }

        return res
    
    
    def __generar_caracteristicas_comparativas(
        self,
        serie_antes:pd.Series,
        serie_despues:pd.Series,
        prefijo:str
    ) -> dict[str, float]:
        """Genera algunas características comparativas entre dos series.

        Args:
            serie_antes (pd.Series): Serie antes del punto de break.
            serie_despues (pd.Series): Serie después del punto de break.
            prefijo (str): Prefijo para el nombre de las características
            que se creen.

        Returns:
            dict[str, float]: Características comparativas generadas. La 
            clave es el nombre de la característica y el valor es el valor
            numérico de la característica calculado.
        """
        
        # Energy distance (para comprobar si las series son similares)
        distance = energy_distance(serie_antes, serie_despues)
        
        # Wasserstein distance (cuanta "masa" hay que mover para que las
        # series sean iguales)
        dist = wasserstein_distance(serie_antes, serie_despues)
        
        # Kolmogorov-Smirnov test (para comprobar si las distribuciones son
        # iguales)
        stat, p_value = ks_2samp(serie_antes, serie_despues)
        
        res = {
            f'{prefijo}_energy_distance': distance,
            f'{prefijo}_wasserstein_distance': dist,
            f'{prefijo}_ks_statistic': stat,
            f'{prefijo}_ks_p_value': p_value
        }
        
        return res