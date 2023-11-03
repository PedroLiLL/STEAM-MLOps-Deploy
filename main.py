#import psutil
from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Instanciamos un objeto de la clase fastapi para construir la aplicación
app = FastAPI(title='STEAM Games: Consultas', description='Esta aplicación permite realizar consultas sobre videojuegos, reseñas de usuarios, recomendaciones y más')

# cargamos las tablas limpias en .parquet
reviews_games = pd.read_parquet('reviews_games.parquet')
steam_games = pd.read_parquet('steam_games_ml.parquet')

# ruta inicial
@app.get("/")
async def index():
    mensaje = 'Bienvenid@ a mi API para consultas sobre videojuegos, reseñas de usuarios y recomendaciones de la plataforma STEAM'
    return {'Mensaje': mensaje}


@app.get("/Play-Time-Genre/{genero}", name="Tiempo de juego por género")
def PlayTimeGenre(genero: str):
    '''
    devuelve el año con mas horas jugadas para dicho género

    Args: 
        genero (str): género del juego

    return:
        dict: diccionario con el género X y el año de lanzamiento con más horas jugadas
    
    '''
    diccionario = {'Action': 2012,
                   'RPG': 2011,
                   'Strategy': 2015,
                   'Casual': 2017,
                   'Simulation': 2006,
                   'Indie': 2013,
                   'Racing': 2016,
                   'Sports': 2013,
                   'Adventure': 2011,
                   'Early Access': 2013,
                   'Free to Play': 2017,
                   'Massively Multiplayer': 2013,
                   'Utilities': 2014,
                   'Animation &amp; Modeling': 2015,
                   'Video Production': 2015,
                   'Design &amp; Illustration': 2012,
                   'Web Publishing': 2012,
                   'Education': 2015,
                   'Software Training': 2015,
                   'Photo Editing': 2015,
                   'Audio Production': 2014}
    
    if genero in diccionario:
        return {f'Año de lanzamiento con más horas jugadas para el género {genero}' : diccionario[genero]}
    else:
        return "No se encontró un género similar al ingresado, ingrese otro tipo de género"


@app.get("/User-For-Genre/{genero}", name="Usuario con mas minutos jugados para un género")
def UserForGenre(genero: str):
    '''
    devuelve el usuario con mas minutos jugados para dicho género

    Args: 
        genero (str): género del juego

    return:
        dict: diccionario con el género X y el usuario con más horas jugadas
    
    '''

    diccionario = {'Action': {'Usuario con más minutos jugados para género Action': 'REBAS_AS_F-T', 'Minutos jugados': [{'Año': 1996, 'Minutos': 0}, {'Año': 1998, 'Minutos': 2}, {'Año': 1999, 'Minutos': 225}, {'Año': 2000, 'Minutos': 0}, {'Año': 2001, 'Minutos': 11}, {'Año': 2002, 'Minutos': 1}, {'Año': 2003, 'Minutos': 1863}, {'Año': 2004, 'Minutos': 2115}, {'Año': 2005, 'Minutos': 3293}, {'Año': 2006, 'Minutos': 138}, {'Año': 2007, 'Minutos': 375}, {'Año': 2008, 'Minutos': 2573}, {'Año': 2009, 'Minutos': 7926}, {'Año': 2010, 'Minutos': 4460}, {'Año': 2011, 'Minutos': 37705}, {'Año': 2012, 'Minutos': 50635}, {'Año': 2013, 'Minutos': 97566}, {'Año': 2014, 'Minutos': 158114}, {'Año': 2015, 'Minutos': 162452}, {'Año': 2016, 'Minutos': 138572}, {'Año': 2017, 'Minutos': 1990}]},
                   'RPG': {'Usuario con más minutos jugados para género RPG': 'REBAS_AS_F-T', 'Minutos jugados': [{'Año': 1999, 'Minutos': 1}, {'Año': 2002, 'Minutos': 0}, {'Año': 2003, 'Minutos': 0}, {'Año': 2005, 'Minutos': 1113}, {'Año': 2007, 'Minutos': 426}, {'Año': 2008, 'Minutos': 30}, {'Año': 2009, 'Minutos': 943}, {'Año': 2010, 'Minutos': 36}, {'Año': 2011, 'Minutos': 19772}, {'Año': 2012, 'Minutos': 14631}, {'Año': 2013, 'Minutos': 31896}, {'Año': 2014, 'Minutos': 81069}, {'Año': 2015, 'Minutos': 94105}, {'Año': 2016, 'Minutos': 91414}, {'Año': 2017, 'Minutos': 55}]},
                   'Strategy': {'Usuario con más minutos jugados para género Strategy': 'Steamified', 'Minutos jugados': [{'Año': 1988, 'Minutos': 0}, {'Año': 1990, 'Minutos': 0}, {'Año': 1993, 'Minutos': 0}, {'Año': 1995, 'Minutos': 2}, {'Año': 1997, 'Minutos': 0}, {'Año': 1998, 'Minutos': 0}, {'Año': 1999, 'Minutos': 0}, {'Año': 2000, 'Minutos': 0}, {'Año': 2001, 'Minutos': 309}, {'Año': 2002, 'Minutos': 75}, {'Año': 2003, 'Minutos': 327}, {'Año': 2004, 'Minutos': 85}, {'Año': 2005, 'Minutos': 502}, {'Año': 2006, 'Minutos': 359}, {'Año': 2007, 'Minutos': 149}, {'Año': 2008, 'Minutos': 1192}, {'Año': 2009, 'Minutos': 2432}, {'Año': 2010, 'Minutos': 1278}, {'Año': 2011, 'Minutos': 5613}, {'Año': 2012, 'Minutos': 4873}, {'Año': 2013, 'Minutos': 10106}, {'Año': 2014, 'Minutos': 36087}, {'Año': 2015, 'Minutos': 67815}, {'Año': 2016, 'Minutos': 59267}, {'Año': 2017, 'Minutos': 1500}]},
                   'Casual': {'Usuario con más minutos jugados para género Casual': 'REBAS_AS_F-T', 'Minutos jugados': [{'Año': 1999, 'Minutos': 0}, {'Año': 2002, 'Minutos': 0}, {'Año': 2007, 'Minutos': 0}, {'Año': 2008, 'Minutos': 1241}, {'Año': 2009, 'Minutos': 2870}, {'Año': 2010, 'Minutos': 5246}, {'Año': 2011, 'Minutos': 16655}, {'Año': 2012, 'Minutos': 19193}, {'Año': 2013, 'Minutos': 30997}, {'Año': 2014, 'Minutos': 72646}, {'Año': 2015, 'Minutos': 112565}, {'Año': 2016, 'Minutos': 111925}, {'Año': 2017, 'Minutos': 58}]},
                   'Simulation': {'Usuario con más minutos jugados para género Simulation': 'REBAS_AS_F-T', 'Minutos jugados': [{'Año': 2006, 'Minutos': 1673}, {'Año': 2007, 'Minutos': 426}, {'Año': 2008, 'Minutos': 388}, {'Año': 2009, 'Minutos': 4642}, {'Año': 2010, 'Minutos': 556}, {'Año': 2011, 'Minutos': 22193}, {'Año': 2012, 'Minutos': 9960}, {'Año': 2013, 'Minutos': 19055}, {'Año': 2014, 'Minutos': 21175}, {'Año': 2015, 'Minutos': 29977}, {'Año': 2016, 'Minutos': 51404}, {'Año': 2017, 'Minutos': 0}]},
                   'Indie': {'Usuario con más minutos jugados para género Indie': 'REBAS_AS_F-T', 'Minutos jugados': [{'Año': 1999, 'Minutos': 0}, {'Año': 2001, 'Minutos': 11}, {'Año': 2003, 'Minutos': 1863}, {'Año': 2005, 'Minutos': 0}, {'Año': 2006, 'Minutos': 1673}, {'Año': 2007, 'Minutos': 0}, {'Año': 2008, 'Minutos': 1366}, {'Año': 2009, 'Minutos': 6395}, {'Año': 2010, 'Minutos': 8239}, {'Año': 2011, 'Minutos': 41078}, {'Año': 2012, 'Minutos': 40191}, {'Año': 2013, 'Minutos': 73006}, {'Año': 2014, 'Minutos': 208536}, {'Año': 2015, 'Minutos': 239255}, {'Año': 2016, 'Minutos': 212445}, {'Año': 2017, 'Minutos': 3096}]},
                   'Racing': {'Usuario con más minutos jugados para género Racing': 'REBAS_AS_F-T', 'Minutos jugados': [{'Año': 2007, 'Minutos': 101}, {'Año': 2008, 'Minutos': 0}, {'Año': 2009, 'Minutos': 467}, {'Año': 2010, 'Minutos': 30}, {'Año': 2011, 'Minutos': 4125}, {'Año': 2012, 'Minutos': 2974}, {'Año': 2013, 'Minutos': 5313}, {'Año': 2014, 'Minutos': 10761}, {'Año': 2015, 'Minutos': 12004}, {'Año': 2016, 'Minutos': 11973}]},
                   'Sports': {'Usuario con más minutos jugados para género Sports': 'REBAS_AS_F-T', 'Minutos jugados': [{'Año': 2009, 'Minutos': 3501}, {'Año': 2011, 'Minutos': 2176}, {'Año': 2013, 'Minutos': 6437}, {'Año': 2014, 'Minutos': 14851}, {'Año': 2015, 'Minutos': 10437}, {'Año': 2016, 'Minutos': 14653}]},
                   'Adventure': {'Usuario con más minutos jugados para género Adventure': 'REBAS_AS_F-T', 'Minutos jugados': [{'Año': 2002, 'Minutos': 0}, {'Año': 2003, 'Minutos': 1863}, {'Año': 2005, 'Minutos': 1113}, {'Año': 2006, 'Minutos': 0}, {'Año': 2007, 'Minutos': 0}, {'Año': 2008, 'Minutos': 1241}, {'Año': 2009, 'Minutos': 2966}, {'Año': 2010, 'Minutos': 717}, {'Año': 2011, 'Minutos': 12619}, {'Año': 2012, 'Minutos': 44997}, {'Año': 2013, 'Minutos': 71001}, {'Año': 2014, 'Minutos': 104096}, {'Año': 2015, 'Minutos': 188300}, {'Año': 2016, 'Minutos': 163831}, {'Año': 2017, 'Minutos': 3555}]},
                   'Early Access': {'Usuario con más minutos jugados para género Early Access': 'Steamified', 'Minutos jugados': [{'Año': 2012, 'Minutos': 111}, {'Año': 2013, 'Minutos': 550}, {'Año': 2014, 'Minutos': 1090}, {'Año': 2015, 'Minutos': 8855}, {'Año': 2016, 'Minutos': 17020}, {'Año': 2017, 'Minutos': 203}, {'Año': 2018, 'Minutos': 296}]},
                   'Free to Play': {'Usuario con más minutos jugados para género Free to Play': 'DownSyndromeKid', 'Minutos jugados': [{'Año': 1996, 'Minutos': 0}, {'Año': 2001, 'Minutos': 2892}, {'Año': 2005, 'Minutos': 0}, {'Año': 2006, 'Minutos': 1104}, {'Año': 2007, 'Minutos': 136}, {'Año': 2008, 'Minutos': 0}, {'Año': 2009, 'Minutos': 0}, {'Año': 2010, 'Minutos': 190}, {'Año': 2011, 'Minutos': 106}, {'Año': 2012, 'Minutos': 29}, {'Año': 2013, 'Minutos': 7302}, {'Año': 2014, 'Minutos': 21264}, {'Año': 2015, 'Minutos': 29678}, {'Año': 2016, 'Minutos': 1598}, {'Año': 2017, 'Minutos': 5856}]},
                   'Massively Multiplayer': {'Usuario con más minutos jugados para género Massively Multiplayer': 'Hati_Hati_Hati_Hati_Hati_Hati', 'Minutos jugados': [{'Año': 2008, 'Minutos': 0}, {'Año': 2009, 'Minutos': 0}, {'Año': 2011, 'Minutos': 265}, {'Año': 2012, 'Minutos': 6681}, {'Año': 2013, 'Minutos': 5274}, {'Año': 2014, 'Minutos': 9794}, {'Año': 2015, 'Minutos': 10827}, {'Año': 2016, 'Minutos': 4893}, {'Año': 2017, 'Minutos': 3191}]},
                   'Utilities': {'Usuario con más minutos jugados para género Utilities': 'NotForPikachu', 'Minutos jugados': [{'Año': 2014, 'Minutos': 46}, {'Año': 2015, 'Minutos': 8665}]},
                   'Animation &amp; Modeling': {'Usuario con más minutos jugados para género Animation &amp; Modeling': '76561198063361762', 'Minutos jugados': [{'Año': 2013, 'Minutos': 4661}]},
                   'Video Production': {'Usuario con más minutos jugados para género Video Production': '76561198126926393', 'Minutos jugados': [{'Año': 2014, 'Minutos': 5223}]},
                   'Design &amp; Illustration': {'Usuario con más minutos jugados para género Design &amp; Illustration': 'H-Alo', 'Minutos jugados': [{'Año': 2012, 'Minutos': 5128}, {'Año': 2015, 'Minutos': 5414}]},
                   'Web Publishing': {'Usuario con más minutos jugados para género Web Publishing': 'H-Alo', 'Minutos jugados': [{'Año': 2005, 'Minutos': 11}, {'Año': 2012, 'Minutos': 5128}, {'Año': 2015, 'Minutos': 5414}]},
                   'Education': {'Usuario con más minutos jugados para género Education': '76561198063361762', 'Minutos jugados': [{'Año': 2013, 'Minutos': 4661}]},
                   'Software Training': {'Usuario con más minutos jugados para género Software Training': 'hterrormc', 'Minutos jugados': [{'Año': 2015, 'Minutos': 4456}]},
                   'Photo Editing': {'Usuario con más minutos jugados para género Photo Editing': 'thugnificent', 'Minutos jugados': [{'Año': 2015, 'Minutos': 0}, {'Año': 2016, 'Minutos': 502}]},
                   'Audio Production': {'Usuario con más minutos jugados para género Audio Production': 'Nyanonymous', 'Minutos jugados': [{'Año': 2014, 'Minutos': 3786}]}}
    
    if genero in diccionario:
        return diccionario[genero]
    else:
        return "No se encontró un género similar al ingresado, ingrese otro tipo de género"


@app.get("/User-Recommend/{anio}", name="Top 3 de juegos MÁS recomendados por usuarios por año")
def UserRecommend(anio: int):
    '''
    devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
    
    Args: 
        anio (int): año de consulta

    return:
        dict: diccionario con los 3 juegos más recomendados por usuarios
        ->  reviews['recommend'] = True(1)
        ->  comentarios positivos(2) o neutros(1)

    '''
    # Si el año de lanzamiento(year_release) no coincide con alguno de los años en los que se hace una reseña(year_posted), se retorna un mensaje de erro
    if anio not in reviews_games['year_posted'].unique():
        return f"Año fuera de rango, ingrese un año válido"
    
    else:
        # Filtramos el dataframe con las filas cuyo año de posteo(year_posted) es mayor o igual al año de publicación(year_release)
        df = reviews_games[reviews_games['year_posted']>=reviews_games['year_release']]
        
        # Filtramos el dataframe 'df' para el año parámetro y la columna sentiment_analysis sea positivo(2) o neutro(1)
        df_anio_dado = df[(df['year_posted']==anio) & (df['sentiment_analysis'].isin([1,2]))]

        # Agrupamos el dataframe 'df_anio_dado' por título del juego ('title'), sumamos las recomendaciones('recommend') para tener los juegos más recomendados y ordenamos de forma descendente
        top = df_anio_dado.groupby('title')['recommend'].sum().sort_values(ascending=False)

        # Construimos el top3
        top3 = [{"Puesto 1": top.index[0]}, {"Puesto 2": top.index[1]}, {"Puesto 3": top.index[2]}]

    return top3


@app.get("/Users-Not-Recommend/{anio}", name="Top 3 de juegos MENOS recomendados por usuarios por año")
def UsersNotRecommend(anio: int):
    '''
    devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado.
    
    Args: 
        anio (int): año de consulta

    return:
        dict: diccionario con los 3 juegos menos recomendados por usuarios
        ->  reviews['recommend'] = False(0)
        ->  comentarios negativos(reviews['sentiment_analysis']==0)

    '''
    # Si el año de lanzamiento(year_release) no coincide con alguno de los años en los que se hace una reseña(year_posted), se retorna un mensaje de erro
    if anio not in reviews_games['year_posted'].unique():
        return f"Año fuera de rango, ingrese un año válido"
    
    else:
        # Filtramos el dataframe con las filas cuyo año de posteo(year_posted) es mayor o igual al año de publicación(year_release)
        df = reviews_games[reviews_games['year_posted']>=reviews_games['year_release']]

        #Ahora filtramos por el año parámetro, recomendaciones negativas(0) y comentarios negativos (0)
        df_anio_dado = df[(df['year_posted']==anio) & (df['recommend']==0) & (df['sentiment_analysis']==0)]

        # Agrupamos respecto al año título del juego, contamos las recomendaciones('recommend') para tener los juegos con más reseñas negtivas y ordenamos de forma descendente
        grupo = df_anio_dado.groupby('title')['recommend'].count().sort_values(ascending=False)

        # Contruimos el top3
        top3 = [{"Puesto 1": grupo.index[0]}, {"Puesto 2": grupo.index[1]}, {"Puesto 3": grupo.index[2]}]

        return top3


@app.get('/sentiment-analysis/{anio}', name='lista con la cantidad de registros de reseñas de usuarios')
def sentiment_analysis(anio: int):
    '''
    Según el año de lanzamiento, devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento

    Args: 
        anio (int): año de lanzamiento (year_release)

    return:
        list: lista con la cantidad de registros de reseñas de usuarios categorizados

    '''
    # Si el año de lanzamiento(year_release) no coincide con alguno de los años en los que se hace una reseña(year_posted), se retorna un mensaje de erro
    if anio not in reviews_games['year_posted'].unique():
        return f"Año fuera de rango, ingrese un año válido"
    
    else:
        # Filtramos el dataframe con las filas cuyo año de posteo(year_posted) es mayor o igual al año de publicación(year_release)
        df = reviews_games[reviews_games['year_posted']>=reviews_games['year_release']]

        # Filtramos el dataframe 'df' para el año parámetro
        df_anio = df[df['year_release'] == anio]

        # Contamos las filas del dataframe 'df_anio' respecto a los valores únicos de la columna 'sentiment_analysis' {0,1,2} y los guardamos en sus respectivos sentimientos
        positivos = df_anio[df_anio['sentiment_analysis']==2].shape[0] # número de filas de sentimientos positivos(2)
        neutros = df_anio[df_anio['sentiment_analysis']==1].shape[0] # número de filas de sentimientos nneutros(1)
        negativos = df_anio[df_anio['sentiment_analysis']==0].shape[0] # número de filas de sentimientos negativos(0)

        return {'Negative': negativos, 'Neutral': neutros, 'Positive': positivos}


# Creamos una instancia de la clase CountVectorizer
vector = CountVectorizer(tokenizer= lambda x: x.split(', '))
# Dividimos cada cadena de descripción en palabras individuales y se crea una matriz de conteo 'matriz_descripcion' que representa cuántas veces aparece cada género en cada videojuego.
matriz_descripcion = vector.fit_transform(steam_games['description'])

@app.get('/Juegos-recomendados/{id_producto}', name='lista con juegos recomendados por juego ingresado')
def recomendacion_juego(id_producto: int):
    '''
    Se ingresa el id de producto (item_id) y retorna una lista con 5 juegos recomendados similares al ingresado (title).
    
    '''
    # Si el id ingresado no se encuentra en la columna de id de la tabla 'steam_games' se le pide al usuario que intente con otro id
    if id_producto not in steam_games['item_id'].values:
        return 'El ID no existe, intente con otro'
    else:
        # buscamos el índice del id ingresado
        index = steam_games.index[steam_games['item_id']==id_producto][0]

        # De la matriz de conteo, tomamos el array de descripciones con índice igual a 'index'
        description_index = matriz_descripcion[index]

        # Calculamos la similitud coseno entre la descripción de entrada y la descripción de las demás filas: cosine_similarity(description_index, matriz_descripcion)
        # Obtenemos los índices de las mayores similitudes mediante el método argsort() y las similitudes ordenadas de manera descendente
        # Tomamos los índices del 1 al 6 [0, 1:6] ya que el índice 0 es el mismo índice de entrada
        indices_maximos = np.argsort(-cosine_similarity(description_index, matriz_descripcion))[0, 1:6]

        # Construimos la lista
        recomendaciones = []
        for i in indices_maximos:
            recomendaciones.append(steam_games['title'][i])
        
        return recomendaciones

#print(psutil.Process().memory_info().rss)