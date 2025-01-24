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
    "subprocess.run"
    "numpy",
    "xml.etree.ElementTree"
    ]

"""Verifica e instala las librerías necesarias."""

revisarLibrerias(required_libraries)

import pandas as pd
from simplekml import Kml
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import subprocess
import sys
import shutil
import math
import numpy as np

def generate_route_from_excel(kml, input_excel_path, hoja_excel, route_points):
    # Leer el archivo Excel
    df = pd.read_excel(input_excel_path,sheet_name=hoja_excel)


    # Agregar puntos al KML
    for index, row in df.iterrows():
        if row['Name'] in route_points:
            coordinates = row['Coordinates']
            lon, lat, *alt = map(float, coordinates.split(","))
            point = kml.newpoint(name=row['Name'], coords=[(lon, lat)])
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
        linestring = kml.newlinestring(name="Route")
        linestring.coords = route_coords
        linestring.style.linestyle.width = 3
        linestring.style.linestyle.color = 'ff0000ff'  # Red line

    return kml

def generate_polygon_from_route(kml, route_points, input_excel_path, hoja_excel, width_meters, line_color, fill_color):
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
    polygon = kml.newpolygon(name="Route Polygon")
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
    
    conjunto_referencia = set(estructuras) - set(conjunto_referencia)
    for estructura in estructuras:
        if estructura in conjunto_referencia:
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
    df = pd.read_excel(input_excel, sheet_name=hoja, usecols="K", skiprows=3, nrows=22)
    listaEstructurasTramo = df.iloc[:, 0].tolist()
    df = pd.read_excel(input_excel, sheet_name=hoja, usecols="L", skiprows=3, nrows=22)
    estructurasTramoLimpias = df.iloc[:, 0].tolist()

    # Imprimir la lista
    print(f'listaEstructurasTramo: {listaEstructurasTramo[0].split(";")}')
    print(f'estructurasTramoLimpias: {estructurasTramoLimpias[0]}')

    for i in range(len(estructurasTramoLimpias)):
        if len(estructurasTramoLimpias[i]) != 0:
            print(f'estructurasTramoLimpias[i]: {estructurasTramoLimpias[i]}')
            estructurasTramoLimpias[i] = estructurasTramoLimpias[i].split(";")


    return listaEstructurasTramo, estructurasTramoLimpias

revisarLibrerias()

# Datos a traer desde excel
input_excel = "MV y PAT Actualizado ChFM - Modificado MJ.xlsx" #*** Nombre se genera desde macro
hoja_excel = "Coordenadas"
estructurasTramo = ["SL_AS-0050", "SL_AS-0051", "SL_AS-0052", "SL_AS-0053", "SL_AS-0054", "SL_AS-0055", "SL_AS-0056", "SL_AS-0057", "SL_AS-0058", "SL_AS-0059", "SL_AS-0060", "SL_AS-0061", "SL_AS-0062", "SL_AS-0063", "SL_AS-0064", "SL_AS-0065", "SL_AS-0066", "SL_AS-0067", "SL_AS-0068", "SL_AS-0069", "SL_AS-0070", "SL_AS-0071", "SL_AS-0072", "SL_AS-0073", "SL_AS-0074", "SL_AS-0075", "SL_AS-0076", "SL_AS-0077", "SL_AS-0078", "SL_AS-0079", "SL_AS-0080", "SL_AS-0081", "SL_AS-0082", "SL_AS-0083", "SL_AS-0084", "SL_AS-0085", "SL_AS-0086", "SL_AS-0087", "SL_AS-0088", "SL_AS-0089", "SL_AS-0090", "SL_AS-0091", "SL_AS-0092", "SL_AS-0093", "SL_AS-0094", "SL_AS-0095", "SL_AS-0096", "SL_AS-0097", "SL_AS-0098", "SL_AS-0099", "SL_AS-0100"]
estructurasTramoLimpias = ["SL_AS-0050", "SL_AS-0051", "SL_AS-0052", "SL_AS-0053", "SL_AS-0054", "SL_AS-0055", "SL_AS-0056", "SL_AS-0057", "SL_AS-0058", "SL_AS-0059",  "SL_AS-0070", "SL_AS-0071", "SL_AS-0072", "SL_AS-0073", "SL_AS-0074", "SL_AS-0075", "SL_AS-0076", "SL_AS-0077", "SL_AS-0078", "SL_AS-0079", "SL_AS-0081", "SL_AS-0082", "SL_AS-0083", "SL_AS-0084", "SL_AS-0085", "SL_AS-0086", "SL_AS-0087", "SL_AS-0088", "SL_AS-0089", "SL_AS-0096", "SL_AS-0097", "SL_AS-0098", "SL_AS-0099", "SL_AS-0100"]
listaEstructurasTramo, estructurasTramoLimpias = obtenerTramos(input_excel)
print(f'listaEstructurasTramo: {listaEstructurasTramo}')
print(f'estructurasTramoLimpias: {estructurasTramoLimpias}')
colorTramoLimpiado = "Verde" #***
colorTramoNoLimpiado = "Rojo" #***
anchoFranja = 100 #Ancho del poligono en metros ***

# Generar los conjuntos
ConjuntosLimpios, ConjuntosNoLimpios = agrupar_conjuntos(estructurasTramo, estructurasTramoLimpias)

if len(ConjuntosLimpios) != 0:
    nombreTramo = ConjuntosLimpios[0][0].split("-")[0]
    kml = Kml() #Genero KML vacio
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
        output_kmz = nombreTramo + "-Limpio.kmz"  # Cambia esto por la ruta de salida del KMZ
        kml = generate_route_from_excel(kml, input_excel, hoja_excel, route_points)
        line_color, fill_color = obtenerColor(colorTramoLimpiado)
        polygon_result = generate_polygon_from_route(kml,route_points, input_excel_path=input_excel, hoja_excel=hoja_excel, width_meters=anchoFranja, line_color = line_color, fill_color = fill_color)
        polygon_result.savekmz(output_kmz) # Guardar el archivo KMZ
        print(f"Archivo KMZ del Tramo Limpiado generado") 
else:
    print("No existen Tramos Limpios")

if len(ConjuntosNoLimpios) != 0:
    kml = Kml() #Genero KML vacio
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
        output_kmz = nombreTramo + "-Limpio.kmz"  # Cambia esto por la ruta de salida del KMZ
        kml = generate_route_from_excel(kml, input_excel, hoja_excel, route_points)
        line_color, fill_color = obtenerColor(colorTramoNoLimpiado)
        polygon_result = generate_polygon_from_route(kml,route_points, input_excel_path=input_excel, hoja_excel=hoja_excel, width_meters=anchoFranja, line_color = line_color, fill_color = fill_color)
        polygon_result.savekmz(output_kmz) # Guardar el archivo KMZ
        print(f"Archivo KMZ del Tramo No Limpiado generado") 
else:
    print("No existen Tramos No Limpios")

