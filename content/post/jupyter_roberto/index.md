---
title: Prueba con html embebido
summary: Trabajaremos con la base de datos de Kaggle "House Prices"
date: 2021-02-01
authors: ["admin"]
tags: ["Regresión"]
featured: true
categories: ["Proyectos"]

---

# Prueba con dataframe as markdown

## 1. Introducción

### 1.1 Presentación del objetivo

En este proyecto vamos a trabajar la predicción de una variable cuantitativa mediante un conjunto de variables tanto cuantitativas como cualitativas. El objetivo en cuentión se trata de predecir el precio de venta de un bien inmueble, y para ello trabajaremos con la famosa base de datos de _kaggle_ _"House Prices - Advanced Regression Techniques"_, que contiene información sobre 1460 propiedades inmobiliarias (casas) vendidas en Ames, Iowa (EEUU). En concreto, tenemos a nuestra disposición 79 variables explicativas que nos describen muchas de las características de las viviendas, más allá de las típicas como el tamaño o la zona residencial. 


### 1.2 Habilidades que trabajaremos

__Ingeniería de atributos__: Una de las grandes ventajas de esta base de datos es que nos va a permitir trabajar de manera intensa la limpieza, transformación y selección de atributos, pues contamos con muchos de los casos que se nos pueden prensentar en un problema de este tipo, como son el trabajar con valores erroneos u omitidos, el manejo de variables altamente correladas o la necesidad de tener que seleccionar variables o reducir la dimensionalidad de nuestra base de datos.

__Visualización de datos__: Para este proyecto vamos a trabajar la visualización de datos con la librería _Seaborn_, que como veremos nos permite realizar gráficos de una manera muy sencilla y con un acabado realmente profesional.

__Tecnicas de regresión__: Finalmente trabajaremos distintas técnicas de regresión para realizar la predicción sobre los datos de test.


### 1.3 Flujo de trabajo 

Vamos a llevar a cabo nuestro proyecto siguiendo el siguiente _workflow_:

1. 



---

## 2. Preprocesamiento de datos y análisis exploratorio 

### 2.1 Carga de las librerías necesarias

Para este proyecto en concreto cargaremos las siguiente librerías:
- Pandas para trabajar con dataframes.
- Seaborn para la realización de gráficos.
- Scikit-learn para modelizar.


```python
import pandas as pd
# import seaborn as sns
```

### 2.2 Carga de los datos y primera aproximación

Lo primero que haremos será cargar los datos en un _dataframe_ de _Pandas_. Sabemos que los datos incluyen una columna llamada "Id" que nos servirá para indexar nuestros datos, así que en la carga le diremos a _Pandas_ que carge esa columna como índice con el atributo _index_col_.


```python
data = pd.read_csv('data/train.csv', index_col='Id')
data.head(1)
```




<html>
<head>
<style>
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
</style>
</head>
<body>
<table style="width:100%" class="dataframe">
    <thead>
        <tr style="text-align: right;">
            <th></th>
            <th style="min-width:120px">MSSubClass</th>
            <th style="width:100%">MSZoning</th>
            <th>LotFrontage</th>
            <th>LotArea</th>
            <th>Street</th>
            <th>Alley</th>
            <th>LotShape</th>
            <th>LandContour</th>
            <th>Utilities</th>
            <th>LotConfig</th>
            <th>LandSlope</th>
            <th>Neighborhood</th>
            <th>Condition1</th>
            <th>Condition2</th>
            <th>BldgType</th>
            <th>HouseStyle</th>
            <th>OverallQual</th>
            <th>OverallCond</th>
            <th>YearBuilt</th>
            <th>YearRemodAdd</th>
            <th>RoofStyle</th>
            <th>RoofMatl</th>
            <th>Exterior1st</th>
            <th>Exterior2nd</th>
            <th>MasVnrType</th>
            <th>MasVnrArea</th>
            <th>ExterQual</th>
            <th>ExterCond</th>
            <th>Foundation</th>
            <th>BsmtQual</th>
            <th>BsmtCond</th>
            <th>BsmtExposure</th>
            <th>BsmtFinType1</th>
            <th>BsmtFinSF1</th>
            <th>BsmtFinType2</th>
            <th>BsmtFinSF2</th>
            <th>BsmtUnfSF</th>
            <th>TotalBsmtSF</th>
            <th>Heating</th>
            <th>HeatingQC</th>
            <th>CentralAir</th>
            <th>Electrical</th>
            <th>1stFlrSF</th>
            <th>2ndFlrSF</th>
            <th>LowQualFinSF</th>
            <th>GrLivArea</th>
            <th>BsmtFullBath</th>
            <th>BsmtHalfBath</th>
            <th>FullBath</th>
            <th>HalfBath</th>
            <th>BedroomAbvGr</th>
            <th>KitchenAbvGr</th>
            <th>KitchenQual</th>
            <th>TotRmsAbvGrd</th>
            <th>Functional</th>
            <th>Fireplaces</th>
            <th>FireplaceQu</th>
            <th>GarageType</th>
            <th>GarageYrBlt</th>
            <th>GarageFinish</th>
            <th>GarageCars</th>
            <th>GarageArea</th>
            <th>GarageQual</th>
            <th>GarageCond</th>
            <th>PavedDrive</th>
            <th>WoodDeckSF</th>
            <th>OpenPorchSF</th>
            <th>EnclosedPorch</th>
            <th>3SsnPorch</th>
            <th>ScreenPorch</th>
            <th>PoolArea</th>
            <th>PoolQC</th>
            <th>Fence</th>
            <th>MiscFeature</th>
            <th>MiscVal</th>
            <th>MoSold</th>
            <th>YrSold</th>
            <th>SaleType</th>
            <th>SaleCondition</th>
            <th>SalePrice</th>
            </tr>
                <tr>
            <th>Id</th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            <th></th>
            </tr>
          </thead>
              <tbody>
                    <tr>
            <th>1</th>
            <td>60</td>
            <td>RL</td>
            <td>65.0</td>
            <td>8450</td>
            <td>Pave</td>
            <td>NaN</td>
            <td>Reg</td>
            <td>Lvl</td>
            <td>AllPub</td>
            <td>Inside</td>
            <td>Gtl</td>
            <td>CollgCr</td>
            <td>Norm</td>
            <td>Norm</td>
            <td>1Fam</td>
            <td>2Story</td>
            <td>7</td>
            <td>5</td>
            <td>2003</td>
            <td>2003</td>
            <td>Gable</td>
            <td>CompShg</td>
            <td>VinylSd</td>
            <td>VinylSd</td>
            <td>BrkFace</td>
            <td>196.0</td>
            <td>Gd</td>
            <td>TA</td>
            <td>PConc</td>
            <td>Gd</td>
            <td>TA</td>
            <td>No</td>
            <td>GLQ</td>
            <td>706</td>
            <td>Unf</td>
            <td>0</td>
            <td>150</td>
            <td>856</td>
            <td>GasA</td>
            <td>Ex</td>
            <td>Y</td>
            <td>SBrkr</td>
            <td>856</td>
            <td>854</td>
            <td>0</td>
            <td>1710</td>
            <td>1</td>
            <td>0</td>
            <td>2</td>
            <td>1</td>
            <td>3</td>
            <td>1</td>
            <td>Gd</td>
            <td>8</td>
            <td>Typ</td>
            <td>0</td>
            <td>NaN</td>
            <td>Attchd</td>
            <td>2003.0</td>
            <td>RFn</td>
            <td>2</td>
            <td>548</td>
            <td>TA</td>
            <td>TA</td>
            <td>Y</td>
            <td>0</td>
            <td>61</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>0</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>NaN</td>
            <td>0</td>
            <td>2</td>
            <td>2008</td>
            <td>WD</td>
            <td>Normal</td>
            <td>208500</td>
            </tr>
          </tbody>
        </table>
</body>
</html>





```python

```
