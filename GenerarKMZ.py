import os

def revisarLibrerias(libraries):
    
    """Verifica e instala las librerías necesarias utilizando os."""
    for library in libraries:
        try:
            __import__(library)  # Intentar importar el módulo
        except ImportError:
            print(f"Instalando {library}...")
            os.system(f"pip install {library}")
        try:
            __import__(library)  # Verificar nuevamente la importación
            print(f"{library} importado correctamente.")
        except ImportError:
            print(f"Error: No se pudo importar {library} después de la instalación.")

# Instalar todas las librerías requeridas
required_libraries = [
    "pandas",
    "simplekml",
    "math",
    "numpy",
    "xml.etree.ElementTree"
    ]

"""Verifica e instala las librerías necesarias."""

revisarLibrerias(required_libraries)


import pandas as pd
from simplekml import Kml
import sys
import math
import numpy as np
from xml.etree import ElementTree as ET
import zipfile

if len(sys.argv) > 1:
    parametro = sys.argv[1]  # Primer argumento después del nombre del script
    print(f"Parámetro recibido: {parametro}")
else:
    print("No se proporcionaron parámetros. Usa: python script.py <parametro>")

def generate_route_from_excel(kml, nombreTramo, input_excel_path, hoja_excel, route_points):
    # Leer el archivo Excel
    df = pd.read_excel(input_excel_path,sheet_name=hoja_excel)
    folder = kml.newfolder(name=nombreTramo + "- Points")  # Nombre de la carpeta en el KML

    # Agregar puntos al KML
    for index, row in df.iterrows():
        if row['Name'] in route_points:
            coordinates = row['Coordinates']
            lon, lat, *alt = map(float, coordinates.split(","))
            point = folder.newpoint(name=row['Name'], coords=[(lon, lat)])
            point.description = row.get('Description', 'No description')

    # Crear una ruta entre los puntos definidos
    route_coords = []
    for point_name in route_points:
        filtered_row = df[df['Name'] == point_name]
        if not filtered_row.empty:
            coordinates = filtered_row.iloc[0]['Coordinates']
            lon, lat, *alt = map(float, coordinates.split(","))
            route_coords.append((lon, lat))

    if route_coords:
        linestring = folder.newlinestring(name="Route")
        linestring.coords = route_coords
        linestring.style.linestyle.width = 3
        linestring.style.linestyle.color = 'ff0000ff'  # Red line

    return kml

def generate_polygon_from_route(kml, nombreTramo, route_points, input_excel_path, hoja_excel, width_meters, line_color, fill_color):
    # Leer el archivo Excel
    df = pd.read_excel(input_excel_path,sheet_name=hoja_excel)

    # Extraer las coordenadas de los puntos de la ruta
    route_coords = []
    for point_name in route_points:
        filtered_row = df[df['Name'] == point_name]
        if not filtered_row.empty:
            coordinates = filtered_row.iloc[0]['Coordinates']
            lon, lat, *alt = map(float, coordinates.split(","))
            route_coords.append((lon, lat))

    if not route_coords:
        raise ValueError("No se encontraron coordenadas para los puntos especificados en route_points.")

    
    # Calcular las coordenadas del polígono
    polygon_coords = []
    for i in range(len(route_coords)):
        lon, lat = route_coords[i]

        # Determinar la dirección del segmento (para calcular el vector perpendicular)
        if i == 0:  # Primer punto
            next_lon, next_lat = route_coords[i + 1]
            dx = next_lon - lon
            dy = next_lat - lat
        elif i == len(route_coords) - 1:  # Último punto
            prev_lon, prev_lat = route_coords[i - 1]
            dx = lon - prev_lon
            dy = lat - prev_lat
        else:  # Puntos intermedios (promedio de las direcciones)
            next_lon, next_lat = route_coords[i + 1]
            prev_lon, prev_lat = route_coords[i - 1]
            dx = (next_lon - prev_lon) / 2
            dy = (next_lat - prev_lat) / 2

        # Normalizar el vector perpendicular
        length = math.sqrt(dx**2 + dy**2)
        perp_dx = -dy / length
        perp_dy = dx / length

        # Conversión de metros a grados:
        # 1 grado de latitud = ~111,320 metros
        # 1 grado de longitud depende de la latitud (cos(lat) * 111,320 metros)
        meters_to_deg_lat = width_meters / 2 / 111320  # Latitud constante
        meters_to_deg_lon = width_meters / 2 / (111320 * math.cos(math.radians(lat)))  # Longitud variable según latitud

        # Agregar puntos desplazados a los lados
        polygon_coords.append((lon + perp_dx * meters_to_deg_lon, lat + perp_dy * meters_to_deg_lat))  # Lado derecho
        polygon_coords.insert(0, (lon - perp_dx * meters_to_deg_lon, lat - perp_dy * meters_to_deg_lat))  # Lado izquierdo

    # Crear el polígono
    folder = kml.newfolder(name=nombreTramo + "_Polygon")  # Nombre de la carpeta en el KML
    polygon = folder.newpolygon(name= nombreTramo + "_Polygon")
    polygon.outerboundaryis.coords = polygon_coords
    polygon.style.polystyle.color = fill_color  # Color de relleno (formato ABGR)
    polygon.style.polystyle.fill = 1  # Activar el relleno
    polygon.style.linestyle.color = line_color  # Color del borde (formato ABGR)
    polygon.style.linestyle.width = 2  # Ancho del borde

    return kml

def generate_route_points(nombreTramo, inicioTramoCompletado, finTramoCompletado):
    """
    Genera una lista de puntos para una ruta basada en el nombre del tramo y el rango de números.
    """
    step = 1 if inicioTramoCompletado <= finTramoCompletado else -1
    return [f"{nombreTramo}-{str(i).zfill(4)}" for i in range(inicioTramoCompletado, finTramoCompletado + step, step)]

def obtenerColor(color):
    if color == "Rojo":
        line_color="ff0000ff"  # Color del borde (RGBA invertido)
        fill_color="880000ff"  # Color de relleno (RGBA invertido)
    if color == "Verde":
        line_color="ff00ff00"  # Color del borde (RGBA invertido)
        fill_color="8800ff00"  # Color de relleno (RGBA invertido)
    if color == "Azul":
        line_color="ffff0000"  # Color del borde (RGBA invertido)
        fill_color="88ff0000"  # Color de relleno (RGBA invertido)
    return line_color, fill_color

# Función para agrupar conjuntos correlativos
def agrupar_conjuntos(estructuras, conjunto_referencia):
    gruposLimpio = []
    grupo_actual = []
    
    for estructura in estructuras:
        if estructura in conjunto_referencia:
            grupo_actual.append(estructura)
        else:
            if grupo_actual:
                gruposLimpio.append(grupo_actual)
                grupo_actual = []
    if grupo_actual:
        gruposLimpio.append(grupo_actual)

    gruposNoLimpio = []
    grupo_actual = []
    
    estructurasNoLimpias = set(estructuras) - set(conjunto_referencia)
    for estructura in estructuras:
        if estructura in estructurasNoLimpias:
            grupo_actual.append(estructura)
        else:
            if grupo_actual:
                gruposNoLimpio.append(grupo_actual)
                grupo_actual = []
    if grupo_actual:
        gruposNoLimpio.append(grupo_actual)

    return gruposLimpio, gruposNoLimpio

def obtenerTramos(input_excel):
    hoja = "Resumen"

    # Leer el rango de celdas específico (columna K, filas 4 a 25)
    df = pd.read_excel(input_excel, sheet_name=hoja, usecols="K", skiprows=2, nrows=22)
    listaEstructurasTramo = df.iloc[:, 0].tolist()
    for i in range(len(listaEstructurasTramo)):
        if str(listaEstructurasTramo[i]) == "nan":
            listaEstructurasTramo[i] = listaEstructurasTramo[i]
        else:
            listaEstructurasTramo[i]  = listaEstructurasTramo[i].split(";")

    df2 = pd.read_excel(input_excel, sheet_name=hoja, usecols="L", skiprows=2, nrows=22)
    listaEstructurasTramoLimpias = df2.iloc[:, 0].tolist()
    for i in range(len(listaEstructurasTramoLimpias)):
        if str(listaEstructurasTramoLimpias[i]) == "nan":
            listaEstructurasTramoLimpias[i]  = listaEstructurasTramoLimpias[i]
        else:
            listaEstructurasTramoLimpias[i] = listaEstructurasTramoLimpias[i].split(";")

    return listaEstructurasTramo, listaEstructurasTramoLimpias



# Datos a traer desde excel
# input_excel = "MV y PAT Actualizado ChFM - Modificado MJ.xlsx" #*** Nombre se genera desde macro
input_excel = parametro
df = pd.read_excel(input_excel, sheet_name="Parámetros") 
rutaActual = df.loc[df['Nombre Parámetro'] == 'Ruta Actual', 'Valor'].values[0]
colorTramoLimpiado = df.loc[df['Nombre Parámetro'] == 'Color Limpio (OK)', 'Valor'].values[0]
colorTramoNoLimpiado = df.loc[df['Nombre Parámetro'] == 'Color No Limpio (P)', 'Valor'].values[0]
anchoFranja = df.loc[df['Nombre Parámetro'] == 'Ancho de franja', 'Valor'].values[0]
hoja_excel = "Coordenadas"
listaEstructurasTramo, listaEstructurasTramoLimpias = obtenerTramos(input_excel)

kml = Kml() #Genero KML vacio

for i in range(len(listaEstructurasTramo)):
    estructurasTramo = listaEstructurasTramo[i]
    estructurasTramoLimpias =listaEstructurasTramoLimpias[i]


    if isinstance(estructurasTramo, list):
        if isinstance(estructurasTramoLimpias, list):
            # Generar los conjuntos
            ConjuntosLimpios, ConjuntosNoLimpios = agrupar_conjuntos(estructurasTramo, estructurasTramoLimpias)

            nombreTramo = ConjuntosLimpios[0][0].split("-")[0]

            output_kmz = nombreTramo + "-Limpio.kmz"  # Cambia esto por la ruta de salida del KMZ
            line_color, fill_color = obtenerColor(colorTramoLimpiado)

            for numeroConjunto in range(len(ConjuntosLimpios)):
                route_points = []
                if ConjuntosLimpios[numeroConjunto][-1] != estructurasTramo[-1]:
                    route_points = np.array(ConjuntosLimpios[numeroConjunto])
                    arrayEstructurasTramo = np.array(estructurasTramo)
                    indexEstructuraSiguiente = np.where(arrayEstructurasTramo == ConjuntosLimpios[numeroConjunto][-1])
                    estructuraSiguiente = estructurasTramo[indexEstructuraSiguiente[0][0] + 1]
                    route_points = np.append(route_points, estructuraSiguiente)
                else:
                    route_points = ConjuntosLimpios[numeroConjunto]
                nombreTramokml = nombreTramo + "-Limpio"
                kml = generate_route_from_excel(kml, nombreTramokml, input_excel, hoja_excel, route_points)
                kml = generate_polygon_from_route(kml, nombreTramokml, route_points, input_excel_path=input_excel, hoja_excel=hoja_excel, width_meters=anchoFranja, line_color = line_color, fill_color = fill_color)

            # kml.savekmz(output_kmz) # Guardar el archivo KMZ
            # print(f"Archivo KMZ del Tramo {nombreTramo} Limpio generado") 

            # kml = Kml() #Genero KML vacio
            output_kmz = nombreTramo + "-No Limpio.kmz"  # Cambia esto por la ruta de salida del KMZ
            line_color, fill_color = obtenerColor(colorTramoNoLimpiado)
            
            for numeroConjunto in range(len(ConjuntosNoLimpios)):
                route_points = []
                if ConjuntosNoLimpios[numeroConjunto][-1] != estructurasTramo[-1]:
                    route_points = np.array(ConjuntosNoLimpios[numeroConjunto])
                    arrayEstructurasTramo = np.array(estructurasTramo)
                    indexEstructuraSiguiente = np.where(arrayEstructurasTramo == ConjuntosNoLimpios[numeroConjunto][-1])
                    estructuraSiguiente = estructurasTramo[indexEstructuraSiguiente[0][0] + 1]
                    route_points = np.append(route_points, estructuraSiguiente)
                else:
                    route_points = ConjuntosNoLimpios[numeroConjunto]
                nombreTramokml = nombreTramo + "-No Limpio"
                kml = generate_route_from_excel(kml, nombreTramokml, input_excel, hoja_excel, route_points)
                kml = generate_polygon_from_route(kml,nombreTramokml, route_points, input_excel_path=input_excel, hoja_excel=hoja_excel, width_meters=anchoFranja, line_color = line_color, fill_color = fill_color)
            
            # print(f"Archivo KMZ del Tramo No Limpio generado")
        else:
            print(f'Todos son No Limpios')
    else:
        print(f'No hay datos cargados de este Tramo')
        continue
# output_kmz = rutaActual + "//" + input_excel.split(".")[0] + ".kmz"
output_kmz = os.path.join(rutaActual, input_excel.split(".")[0] + ".kmz")

kml.savekmz(output_kmz) # Guardar el archivo KMZ


# #Consolidar kmz's
# kmz_files = [f for f in os.listdir() if f.endswith('.kmz')]
# nombreKMZOut = input_excel.split(".")[0] + ".kmz"
# merge_kmz(kmz_files, )
