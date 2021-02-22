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
print(data.head().to_markdown())
```

    |   Id |   MSSubClass | MSZoning   |   LotFrontage |   LotArea | Street   |   Alley | LotShape   | LandContour   | Utilities   | LotConfig   | LandSlope   | Neighborhood   | Condition1   | Condition2   | BldgType   | HouseStyle   |   OverallQual |   OverallCond |   YearBuilt |   YearRemodAdd | RoofStyle   | RoofMatl   | Exterior1st   | Exterior2nd   | MasVnrType   |   MasVnrArea | ExterQual   | ExterCond   | Foundation   | BsmtQual   | BsmtCond   | BsmtExposure   | BsmtFinType1   |   BsmtFinSF1 | BsmtFinType2   |   BsmtFinSF2 |   BsmtUnfSF |   TotalBsmtSF | Heating   | HeatingQC   | CentralAir   | Electrical   |   1stFlrSF |   2ndFlrSF |   LowQualFinSF |   GrLivArea |   BsmtFullBath |   BsmtHalfBath |   FullBath |   HalfBath |   BedroomAbvGr |   KitchenAbvGr | KitchenQual   |   TotRmsAbvGrd | Functional   |   Fireplaces | FireplaceQu   | GarageType   |   GarageYrBlt | GarageFinish   |   GarageCars |   GarageArea | GarageQual   | GarageCond   | PavedDrive   |   WoodDeckSF |   OpenPorchSF |   EnclosedPorch |   3SsnPorch |   ScreenPorch |   PoolArea |   PoolQC |   Fence |   MiscFeature |   MiscVal |   MoSold |   YrSold | SaleType   | SaleCondition   |   SalePrice |
    |-----:|-------------:|:-----------|--------------:|----------:|:---------|--------:|:-----------|:--------------|:------------|:------------|:------------|:---------------|:-------------|:-------------|:-----------|:-------------|--------------:|--------------:|------------:|---------------:|:------------|:-----------|:--------------|:--------------|:-------------|-------------:|:------------|:------------|:-------------|:-----------|:-----------|:---------------|:---------------|-------------:|:---------------|-------------:|------------:|--------------:|:----------|:------------|:-------------|:-------------|-----------:|-----------:|---------------:|------------:|---------------:|---------------:|-----------:|-----------:|---------------:|---------------:|:--------------|---------------:|:-------------|-------------:|:--------------|:-------------|--------------:|:---------------|-------------:|-------------:|:-------------|:-------------|:-------------|-------------:|--------------:|----------------:|------------:|--------------:|-----------:|---------:|--------:|--------------:|----------:|---------:|---------:|:-----------|:----------------|------------:|
    |    1 |           60 | RL         |            65 |      8450 | Pave     |     nan | Reg        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        2003 |           2003 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          196 | Gd          | TA          | PConc        | Gd         | TA         | No             | GLQ            |          706 | Unf            |            0 |         150 |           856 | GasA      | Ex          | Y            | SBrkr        |        856 |        854 |              0 |        1710 |              1 |              0 |          2 |          1 |              3 |              1 | Gd            |              8 | Typ          |            0 | nan           | Attchd       |          2003 | RFn            |            2 |          548 | TA           | TA           | Y            |            0 |            61 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        2 |     2008 | WD         | Normal          |      208500 |
    |    2 |           20 | RL         |            80 |      9600 | Pave     |     nan | Reg        | Lvl           | AllPub      | FR2         | Gtl         | Veenker        | Feedr        | Norm         | 1Fam       | 1Story       |             6 |             8 |        1976 |           1976 | Gable       | CompShg    | MetalSd       | MetalSd       | None         |            0 | TA          | TA          | CBlock       | Gd         | TA         | Gd             | ALQ            |          978 | Unf            |            0 |         284 |          1262 | GasA      | Ex          | Y            | SBrkr        |       1262 |          0 |              0 |        1262 |              0 |              1 |          2 |          0 |              3 |              1 | TA            |              6 | Typ          |            1 | TA            | Attchd       |          1976 | RFn            |            2 |          460 | TA           | TA           | Y            |          298 |             0 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        5 |     2007 | WD         | Normal          |      181500 |
    |    3 |           60 | RL         |            68 |     11250 | Pave     |     nan | IR1        | Lvl           | AllPub      | Inside      | Gtl         | CollgCr        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        2001 |           2002 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          162 | Gd          | TA          | PConc        | Gd         | TA         | Mn             | GLQ            |          486 | Unf            |            0 |         434 |           920 | GasA      | Ex          | Y            | SBrkr        |        920 |        866 |              0 |        1786 |              1 |              0 |          2 |          1 |              3 |              1 | Gd            |              6 | Typ          |            1 | TA            | Attchd       |          2001 | RFn            |            2 |          608 | TA           | TA           | Y            |            0 |            42 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        9 |     2008 | WD         | Normal          |      223500 |
    |    4 |           70 | RL         |            60 |      9550 | Pave     |     nan | IR1        | Lvl           | AllPub      | Corner      | Gtl         | Crawfor        | Norm         | Norm         | 1Fam       | 2Story       |             7 |             5 |        1915 |           1970 | Gable       | CompShg    | Wd Sdng       | Wd Shng       | None         |            0 | TA          | TA          | BrkTil       | TA         | Gd         | No             | ALQ            |          216 | Unf            |            0 |         540 |           756 | GasA      | Gd          | Y            | SBrkr        |        961 |        756 |              0 |        1717 |              1 |              0 |          1 |          0 |              3 |              1 | Gd            |              7 | Typ          |            1 | Gd            | Detchd       |          1998 | Unf            |            3 |          642 | TA           | TA           | Y            |            0 |            35 |             272 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |        2 |     2006 | WD         | Abnorml         |      140000 |
    |    5 |           60 | RL         |            84 |     14260 | Pave     |     nan | IR1        | Lvl           | AllPub      | FR2         | Gtl         | NoRidge        | Norm         | Norm         | 1Fam       | 2Story       |             8 |             5 |        2000 |           2000 | Gable       | CompShg    | VinylSd       | VinylSd       | BrkFace      |          350 | Gd          | TA          | PConc        | Gd         | TA         | Av             | GLQ            |          655 | Unf            |            0 |         490 |          1145 | GasA      | Ex          | Y            | SBrkr        |       1145 |       1053 |              0 |        2198 |              1 |              0 |          2 |          1 |              4 |              1 | Gd            |              9 | Typ          |            1 | TA            | Attchd       |          2000 | RFn            |            3 |          836 | TA           | TA           | Y            |          192 |            84 |               0 |           0 |             0 |          0 |      nan |     nan |           nan |         0 |       12 |     2008 | WD         | Normal          |      250000 |



```python
import datatable as dt
```


<style type='text/css'>
.datatable table.frame { margin-bottom: 0; }
.datatable table.frame thead { border-bottom: none; }
.datatable table.frame tr.coltypes td {  color: #FFFFFF;  line-height: 6px;  padding: 0 0.5em;}
.datatable .bool    { background: #DDDD99; }
.datatable .object  { background: #565656; }
.datatable .int     { background: #5D9E5D; }
.datatable .float   { background: #4040CC; }
.datatable .str     { background: #CC4040; }
.datatable .row_index {  background: var(--jp-border-color3);  border-right: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  font-size: 9px;}
.datatable .frame tr.coltypes .row_index {  background: var(--jp-border-color0);}
.datatable th:nth-child(2) { padding-left: 12px; }
.datatable .hellipsis {  color: var(--jp-cell-editor-border-color);}
.datatable .vellipsis {  background: var(--jp-layout-color0);  color: var(--jp-cell-editor-border-color);}
.datatable .na {  color: var(--jp-cell-editor-border-color);  font-size: 80%;}
.datatable .footer { font-size: 9px; }
.datatable .frame_dimensions {  background: var(--jp-border-color3);  border-top: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  display: inline-block;  opacity: 0.6;  padding: 1px 10px 1px 5px;}
</style>




```python
dt_data = dt.fread('data/train.csv')
```


```python
dt_data.head()
```




<div class='datatable'>
  <table class='frame'>
  <thead>
    <tr class='colnames'><td class='row_index'></td><th>Id</th><th>MSSubClass</th><th>MSZoning</th><th>LotFrontage</th><th>LotArea</th><th>Street</th><th>Alley</th><th>LotShape</th><th>LandContour</th><th>Utilities</th><th class='vellipsis'>&hellip;</th><th>MoSold</th><th>YrSold</th><th>SaleType</th><th>SaleCondition</th><th>SalePrice</th></tr>
    <tr class='coltypes'><td class='row_index'></td><td class='int' title='int32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='int' title='int32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='str' title='str32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='int' title='int32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='int' title='int32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='str' title='str32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='str' title='str32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='str' title='str32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='str' title='str32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='str' title='str32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td></td><td class='int' title='int32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='int' title='int32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='str' title='str32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='str' title='str32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td><td class='int' title='int32'>&#x25AA;&#x25AA;&#x25AA;&#x25AA;</td></tr>
  </thead>
  <tbody>
    <tr><td class='row_index'>0</td><td>1</td><td>60</td><td>RL</td><td>65</td><td>8450</td><td>Pave</td><td><span class=na>NA</span></td><td>Reg</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>2</td><td>2008</td><td>WD</td><td>Normal</td><td>208500</td></tr>
    <tr><td class='row_index'>1</td><td>2</td><td>20</td><td>RL</td><td>80</td><td>9600</td><td>Pave</td><td><span class=na>NA</span></td><td>Reg</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>5</td><td>2007</td><td>WD</td><td>Normal</td><td>181500</td></tr>
    <tr><td class='row_index'>2</td><td>3</td><td>60</td><td>RL</td><td>68</td><td>11250</td><td>Pave</td><td><span class=na>NA</span></td><td>IR1</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>9</td><td>2008</td><td>WD</td><td>Normal</td><td>223500</td></tr>
    <tr><td class='row_index'>3</td><td>4</td><td>70</td><td>RL</td><td>60</td><td>9550</td><td>Pave</td><td><span class=na>NA</span></td><td>IR1</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>2</td><td>2006</td><td>WD</td><td>Abnorml</td><td>140000</td></tr>
    <tr><td class='row_index'>4</td><td>5</td><td>60</td><td>RL</td><td>84</td><td>14260</td><td>Pave</td><td><span class=na>NA</span></td><td>IR1</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>12</td><td>2008</td><td>WD</td><td>Normal</td><td>250000</td></tr>
    <tr><td class='row_index'>5</td><td>6</td><td>50</td><td>RL</td><td>85</td><td>14115</td><td>Pave</td><td><span class=na>NA</span></td><td>IR1</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>10</td><td>2009</td><td>WD</td><td>Normal</td><td>143000</td></tr>
    <tr><td class='row_index'>6</td><td>7</td><td>20</td><td>RL</td><td>75</td><td>10084</td><td>Pave</td><td><span class=na>NA</span></td><td>Reg</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>8</td><td>2007</td><td>WD</td><td>Normal</td><td>307000</td></tr>
    <tr><td class='row_index'>7</td><td>8</td><td>60</td><td>RL</td><td><span class=na>NA</span></td><td>10382</td><td>Pave</td><td><span class=na>NA</span></td><td>IR1</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>11</td><td>2009</td><td>WD</td><td>Normal</td><td>200000</td></tr>
    <tr><td class='row_index'>8</td><td>9</td><td>50</td><td>RM</td><td>51</td><td>6120</td><td>Pave</td><td><span class=na>NA</span></td><td>Reg</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>4</td><td>2008</td><td>WD</td><td>Abnorml</td><td>129900</td></tr>
    <tr><td class='row_index'>9</td><td>10</td><td>190</td><td>RL</td><td>50</td><td>7420</td><td>Pave</td><td><span class=na>NA</span></td><td>Reg</td><td>Lvl</td><td>AllPub</td><td class=vellipsis>&hellip;</td><td>1</td><td>2008</td><td>WD</td><td>Normal</td><td>118000</td></tr>
  </tbody>
  </table>
  <div class='footer'>
    <div class='frame_dimensions'>10 rows &times; 81 columns</div>
  </div>
</div>





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
