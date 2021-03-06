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
data.head(1).to_html()
```




    '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>MSSubClass</th>\n      <th>MSZoning</th>\n      <th>LotFrontage</th>\n      <th>LotArea</th>\n      <th>Street</th>\n      <th>Alley</th>\n      <th>LotShape</th>\n      <th>LandContour</th>\n      <th>Utilities</th>\n      <th>LotConfig</th>\n      <th>LandSlope</th>\n      <th>Neighborhood</th>\n      <th>Condition1</th>\n      <th>Condition2</th>\n      <th>BldgType</th>\n      <th>HouseStyle</th>\n      <th>OverallQual</th>\n      <th>OverallCond</th>\n      <th>YearBuilt</th>\n      <th>YearRemodAdd</th>\n      <th>RoofStyle</th>\n      <th>RoofMatl</th>\n      <th>Exterior1st</th>\n      <th>Exterior2nd</th>\n      <th>MasVnrType</th>\n      <th>MasVnrArea</th>\n      <th>ExterQual</th>\n      <th>ExterCond</th>\n      <th>Foundation</th>\n      <th>BsmtQual</th>\n      <th>BsmtCond</th>\n      <th>BsmtExposure</th>\n      <th>BsmtFinType1</th>\n      <th>BsmtFinSF1</th>\n      <th>BsmtFinType2</th>\n      <th>BsmtFinSF2</th>\n      <th>BsmtUnfSF</th>\n      <th>TotalBsmtSF</th>\n      <th>Heating</th>\n      <th>HeatingQC</th>\n      <th>CentralAir</th>\n      <th>Electrical</th>\n      <th>1stFlrSF</th>\n      <th>2ndFlrSF</th>\n      <th>LowQualFinSF</th>\n      <th>GrLivArea</th>\n      <th>BsmtFullBath</th>\n      <th>BsmtHalfBath</th>\n      <th>FullBath</th>\n      <th>HalfBath</th>\n      <th>BedroomAbvGr</th>\n      <th>KitchenAbvGr</th>\n      <th>KitchenQual</th>\n      <th>TotRmsAbvGrd</th>\n      <th>Functional</th>\n      <th>Fireplaces</th>\n      <th>FireplaceQu</th>\n      <th>GarageType</th>\n      <th>GarageYrBlt</th>\n      <th>GarageFinish</th>\n      <th>GarageCars</th>\n      <th>GarageArea</th>\n      <th>GarageQual</th>\n      <th>GarageCond</th>\n      <th>PavedDrive</th>\n      <th>WoodDeckSF</th>\n      <th>OpenPorchSF</th>\n      <th>EnclosedPorch</th>\n      <th>3SsnPorch</th>\n      <th>ScreenPorch</th>\n      <th>PoolArea</th>\n      <th>PoolQC</th>\n      <th>Fence</th>\n      <th>MiscFeature</th>\n      <th>MiscVal</th>\n      <th>MoSold</th>\n      <th>YrSold</th>\n      <th>SaleType</th>\n      <th>SaleCondition</th>\n      <th>SalePrice</th>\n    </tr>\n    <tr>\n      <th>Id</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>60</td>\n      <td>RL</td>\n      <td>65.0</td>\n      <td>8450</td>\n      <td>Pave</td>\n      <td>NaN</td>\n      <td>Reg</td>\n      <td>Lvl</td>\n      <td>AllPub</td>\n      <td>Inside</td>\n      <td>Gtl</td>\n      <td>CollgCr</td>\n      <td>Norm</td>\n      <td>Norm</td>\n      <td>1Fam</td>\n      <td>2Story</td>\n      <td>7</td>\n      <td>5</td>\n      <td>2003</td>\n      <td>2003</td>\n      <td>Gable</td>\n      <td>CompShg</td>\n      <td>VinylSd</td>\n      <td>VinylSd</td>\n      <td>BrkFace</td>\n      <td>196.0</td>\n      <td>Gd</td>\n      <td>TA</td>\n      <td>PConc</td>\n      <td>Gd</td>\n      <td>TA</td>\n      <td>No</td>\n      <td>GLQ</td>\n      <td>706</td>\n      <td>Unf</td>\n      <td>0</td>\n      <td>150</td>\n      <td>856</td>\n      <td>GasA</td>\n      <td>Ex</td>\n      <td>Y</td>\n      <td>SBrkr</td>\n      <td>856</td>\n      <td>854</td>\n      <td>0</td>\n      <td>1710</td>\n      <td>1</td>\n      <td>0</td>\n      <td>2</td>\n      <td>1</td>\n      <td>3</td>\n      <td>1</td>\n      <td>Gd</td>\n      <td>8</td>\n      <td>Typ</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>Attchd</td>\n      <td>2003.0</td>\n      <td>RFn</td>\n      <td>2</td>\n      <td>548</td>\n      <td>TA</td>\n      <td>TA</td>\n      <td>Y</td>\n      <td>0</td>\n      <td>61</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>0</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>2</td>\n      <td>2008</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>208500</td>\n    </tr>\n  </tbody>\n</table>'



```html
<table border="1" class="dataframe">
    <thead>
        <tr style="text-align: right;">
            <th></th>
            <th>MSSubClass</th>
            <th>MSZoning</th>
            </tr>
        <tr>
            <th>Id</th>
            <th></th>
            <th></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>1</th>
            <td>60</td>
            <td>RL</td>
        </tr>
    </tbody>
</table>'
```


```python

```
