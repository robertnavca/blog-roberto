---
title: Predicción del precio de venta de la propiedad
summary: Trabajaremos con la base de datos de Kaggle "House Prices"
date: 2021-02-01
authors: ["admin"]
tags: ["Regresión"]
featured: true
categories: ["Proyectos"]

---

# Predicción del precio de venta de la propiedad

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
print(data.head().to_html())
```

    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>MSSubClass</th>
          <th>MSZoning</th>
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
        <tr>
          <th>2</th>
          <td>20</td>
          <td>RL</td>
          <td>80.0</td>
          <td>9600</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>Reg</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>FR2</td>
          <td>Gtl</td>
          <td>Veenker</td>
          <td>Feedr</td>
          <td>Norm</td>
          <td>1Fam</td>
          <td>1Story</td>
          <td>6</td>
          <td>8</td>
          <td>1976</td>
          <td>1976</td>
          <td>Gable</td>
          <td>CompShg</td>
          <td>MetalSd</td>
          <td>MetalSd</td>
          <td>None</td>
          <td>0.0</td>
          <td>TA</td>
          <td>TA</td>
          <td>CBlock</td>
          <td>Gd</td>
          <td>TA</td>
          <td>Gd</td>
          <td>ALQ</td>
          <td>978</td>
          <td>Unf</td>
          <td>0</td>
          <td>284</td>
          <td>1262</td>
          <td>GasA</td>
          <td>Ex</td>
          <td>Y</td>
          <td>SBrkr</td>
          <td>1262</td>
          <td>0</td>
          <td>0</td>
          <td>1262</td>
          <td>0</td>
          <td>1</td>
          <td>2</td>
          <td>0</td>
          <td>3</td>
          <td>1</td>
          <td>TA</td>
          <td>6</td>
          <td>Typ</td>
          <td>1</td>
          <td>TA</td>
          <td>Attchd</td>
          <td>1976.0</td>
          <td>RFn</td>
          <td>2</td>
          <td>460</td>
          <td>TA</td>
          <td>TA</td>
          <td>Y</td>
          <td>298</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>5</td>
          <td>2007</td>
          <td>WD</td>
          <td>Normal</td>
          <td>181500</td>
        </tr>
        <tr>
          <th>3</th>
          <td>60</td>
          <td>RL</td>
          <td>68.0</td>
          <td>11250</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
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
          <td>2001</td>
          <td>2002</td>
          <td>Gable</td>
          <td>CompShg</td>
          <td>VinylSd</td>
          <td>VinylSd</td>
          <td>BrkFace</td>
          <td>162.0</td>
          <td>Gd</td>
          <td>TA</td>
          <td>PConc</td>
          <td>Gd</td>
          <td>TA</td>
          <td>Mn</td>
          <td>GLQ</td>
          <td>486</td>
          <td>Unf</td>
          <td>0</td>
          <td>434</td>
          <td>920</td>
          <td>GasA</td>
          <td>Ex</td>
          <td>Y</td>
          <td>SBrkr</td>
          <td>920</td>
          <td>866</td>
          <td>0</td>
          <td>1786</td>
          <td>1</td>
          <td>0</td>
          <td>2</td>
          <td>1</td>
          <td>3</td>
          <td>1</td>
          <td>Gd</td>
          <td>6</td>
          <td>Typ</td>
          <td>1</td>
          <td>TA</td>
          <td>Attchd</td>
          <td>2001.0</td>
          <td>RFn</td>
          <td>2</td>
          <td>608</td>
          <td>TA</td>
          <td>TA</td>
          <td>Y</td>
          <td>0</td>
          <td>42</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>9</td>
          <td>2008</td>
          <td>WD</td>
          <td>Normal</td>
          <td>223500</td>
        </tr>
        <tr>
          <th>4</th>
          <td>70</td>
          <td>RL</td>
          <td>60.0</td>
          <td>9550</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>Corner</td>
          <td>Gtl</td>
          <td>Crawfor</td>
          <td>Norm</td>
          <td>Norm</td>
          <td>1Fam</td>
          <td>2Story</td>
          <td>7</td>
          <td>5</td>
          <td>1915</td>
          <td>1970</td>
          <td>Gable</td>
          <td>CompShg</td>
          <td>Wd Sdng</td>
          <td>Wd Shng</td>
          <td>None</td>
          <td>0.0</td>
          <td>TA</td>
          <td>TA</td>
          <td>BrkTil</td>
          <td>TA</td>
          <td>Gd</td>
          <td>No</td>
          <td>ALQ</td>
          <td>216</td>
          <td>Unf</td>
          <td>0</td>
          <td>540</td>
          <td>756</td>
          <td>GasA</td>
          <td>Gd</td>
          <td>Y</td>
          <td>SBrkr</td>
          <td>961</td>
          <td>756</td>
          <td>0</td>
          <td>1717</td>
          <td>1</td>
          <td>0</td>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>1</td>
          <td>Gd</td>
          <td>7</td>
          <td>Typ</td>
          <td>1</td>
          <td>Gd</td>
          <td>Detchd</td>
          <td>1998.0</td>
          <td>Unf</td>
          <td>3</td>
          <td>642</td>
          <td>TA</td>
          <td>TA</td>
          <td>Y</td>
          <td>0</td>
          <td>35</td>
          <td>272</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>2</td>
          <td>2006</td>
          <td>WD</td>
          <td>Abnorml</td>
          <td>140000</td>
        </tr>
        <tr>
          <th>5</th>
          <td>60</td>
          <td>RL</td>
          <td>84.0</td>
          <td>14260</td>
          <td>Pave</td>
          <td>NaN</td>
          <td>IR1</td>
          <td>Lvl</td>
          <td>AllPub</td>
          <td>FR2</td>
          <td>Gtl</td>
          <td>NoRidge</td>
          <td>Norm</td>
          <td>Norm</td>
          <td>1Fam</td>
          <td>2Story</td>
          <td>8</td>
          <td>5</td>
          <td>2000</td>
          <td>2000</td>
          <td>Gable</td>
          <td>CompShg</td>
          <td>VinylSd</td>
          <td>VinylSd</td>
          <td>BrkFace</td>
          <td>350.0</td>
          <td>Gd</td>
          <td>TA</td>
          <td>PConc</td>
          <td>Gd</td>
          <td>TA</td>
          <td>Av</td>
          <td>GLQ</td>
          <td>655</td>
          <td>Unf</td>
          <td>0</td>
          <td>490</td>
          <td>1145</td>
          <td>GasA</td>
          <td>Ex</td>
          <td>Y</td>
          <td>SBrkr</td>
          <td>1145</td>
          <td>1053</td>
          <td>0</td>
          <td>2198</td>
          <td>1</td>
          <td>0</td>
          <td>2</td>
          <td>1</td>
          <td>4</td>
          <td>1</td>
          <td>Gd</td>
          <td>9</td>
          <td>Typ</td>
          <td>1</td>
          <td>TA</td>
          <td>Attchd</td>
          <td>2000.0</td>
          <td>RFn</td>
          <td>3</td>
          <td>836</td>
          <td>TA</td>
          <td>TA</td>
          <td>Y</td>
          <td>192</td>
          <td>84</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0</td>
          <td>12</td>
          <td>2008</td>
          <td>WD</td>
          <td>Normal</td>
          <td>250000</td>
        </tr>
      </tbody>
    </table>



```python
display(data.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
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
      <td>...</td>
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
    <tr>
      <th>2</th>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



```python
print(data.head())
```

        MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape  \
    Id                                                                    
    1           60       RL         65.0     8450   Pave   NaN      Reg   
    2           20       RL         80.0     9600   Pave   NaN      Reg   
    3           60       RL         68.0    11250   Pave   NaN      IR1   
    4           70       RL         60.0     9550   Pave   NaN      IR1   
    5           60       RL         84.0    14260   Pave   NaN      IR1   
    
       LandContour Utilities LotConfig  ... PoolArea PoolQC Fence MiscFeature  \
    Id                                  ...                                     
    1          Lvl    AllPub    Inside  ...        0    NaN   NaN         NaN   
    2          Lvl    AllPub       FR2  ...        0    NaN   NaN         NaN   
    3          Lvl    AllPub    Inside  ...        0    NaN   NaN         NaN   
    4          Lvl    AllPub    Corner  ...        0    NaN   NaN         NaN   
    5          Lvl    AllPub       FR2  ...        0    NaN   NaN         NaN   
    
       MiscVal MoSold  YrSold  SaleType  SaleCondition  SalePrice  
    Id                                                             
    1        0      2    2008        WD         Normal     208500  
    2        0      5    2007        WD         Normal     181500  
    3        0      9    2008        WD         Normal     223500  
    4        0      2    2006        WD        Abnorml     140000  
    5        0     12    2008        WD         Normal     250000  
    
    [5 rows x 80 columns]



```python
pd.options.display.html.table_schema=True
```


```python
display(data.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>...</th>
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
      <td>...</td>
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
    <tr>
      <th>2</th>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



```python

```
