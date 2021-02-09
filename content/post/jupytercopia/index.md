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
data.head()
```




<style  type="text/css" >
</style><table id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MSSubClass</th>        <th class="col_heading level0 col1" >MSZoning</th>        <th class="col_heading level0 col2" >LotFrontage</th>        <th class="col_heading level0 col3" >LotArea</th>        <th class="col_heading level0 col4" >Street</th>        <th class="col_heading level0 col5" >Alley</th>        <th class="col_heading level0 col6" >LotShape</th>        <th class="col_heading level0 col7" >LandContour</th>        <th class="col_heading level0 col8" >Utilities</th>        <th class="col_heading level0 col9" >LotConfig</th>        <th class="col_heading level0 col10" >LandSlope</th>        <th class="col_heading level0 col11" >Neighborhood</th>        <th class="col_heading level0 col12" >Condition1</th>        <th class="col_heading level0 col13" >Condition2</th>        <th class="col_heading level0 col14" >BldgType</th>        <th class="col_heading level0 col15" >HouseStyle</th>        <th class="col_heading level0 col16" >OverallQual</th>        <th class="col_heading level0 col17" >OverallCond</th>        <th class="col_heading level0 col18" >YearBuilt</th>        <th class="col_heading level0 col19" >YearRemodAdd</th>        <th class="col_heading level0 col20" >RoofStyle</th>        <th class="col_heading level0 col21" >RoofMatl</th>        <th class="col_heading level0 col22" >Exterior1st</th>        <th class="col_heading level0 col23" >Exterior2nd</th>        <th class="col_heading level0 col24" >MasVnrType</th>        <th class="col_heading level0 col25" >MasVnrArea</th>        <th class="col_heading level0 col26" >ExterQual</th>        <th class="col_heading level0 col27" >ExterCond</th>        <th class="col_heading level0 col28" >Foundation</th>        <th class="col_heading level0 col29" >BsmtQual</th>        <th class="col_heading level0 col30" >BsmtCond</th>        <th class="col_heading level0 col31" >BsmtExposure</th>        <th class="col_heading level0 col32" >BsmtFinType1</th>        <th class="col_heading level0 col33" >BsmtFinSF1</th>        <th class="col_heading level0 col34" >BsmtFinType2</th>        <th class="col_heading level0 col35" >BsmtFinSF2</th>        <th class="col_heading level0 col36" >BsmtUnfSF</th>        <th class="col_heading level0 col37" >TotalBsmtSF</th>        <th class="col_heading level0 col38" >Heating</th>        <th class="col_heading level0 col39" >HeatingQC</th>        <th class="col_heading level0 col40" >CentralAir</th>        <th class="col_heading level0 col41" >Electrical</th>        <th class="col_heading level0 col42" >1stFlrSF</th>        <th class="col_heading level0 col43" >2ndFlrSF</th>        <th class="col_heading level0 col44" >LowQualFinSF</th>        <th class="col_heading level0 col45" >GrLivArea</th>        <th class="col_heading level0 col46" >BsmtFullBath</th>        <th class="col_heading level0 col47" >BsmtHalfBath</th>        <th class="col_heading level0 col48" >FullBath</th>        <th class="col_heading level0 col49" >HalfBath</th>        <th class="col_heading level0 col50" >BedroomAbvGr</th>        <th class="col_heading level0 col51" >KitchenAbvGr</th>        <th class="col_heading level0 col52" >KitchenQual</th>        <th class="col_heading level0 col53" >TotRmsAbvGrd</th>        <th class="col_heading level0 col54" >Functional</th>        <th class="col_heading level0 col55" >Fireplaces</th>        <th class="col_heading level0 col56" >FireplaceQu</th>        <th class="col_heading level0 col57" >GarageType</th>        <th class="col_heading level0 col58" >GarageYrBlt</th>        <th class="col_heading level0 col59" >GarageFinish</th>        <th class="col_heading level0 col60" >GarageCars</th>        <th class="col_heading level0 col61" >GarageArea</th>        <th class="col_heading level0 col62" >GarageQual</th>        <th class="col_heading level0 col63" >GarageCond</th>        <th class="col_heading level0 col64" >PavedDrive</th>        <th class="col_heading level0 col65" >WoodDeckSF</th>        <th class="col_heading level0 col66" >OpenPorchSF</th>        <th class="col_heading level0 col67" >EnclosedPorch</th>        <th class="col_heading level0 col68" >3SsnPorch</th>        <th class="col_heading level0 col69" >ScreenPorch</th>        <th class="col_heading level0 col70" >PoolArea</th>        <th class="col_heading level0 col71" >PoolQC</th>        <th class="col_heading level0 col72" >Fence</th>        <th class="col_heading level0 col73" >MiscFeature</th>        <th class="col_heading level0 col74" >MiscVal</th>        <th class="col_heading level0 col75" >MoSold</th>        <th class="col_heading level0 col76" >YrSold</th>        <th class="col_heading level0 col77" >SaleType</th>        <th class="col_heading level0 col78" >SaleCondition</th>        <th class="col_heading level0 col79" >SalePrice</th>    </tr>    <tr>        <th class="index_name level0" >Id</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5level0_row0" class="row_heading level0 row0" >1</th>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col0" class="data row0 col0" >60</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col1" class="data row0 col1" >RL</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col2" class="data row0 col2" >65.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col3" class="data row0 col3" >8450</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col4" class="data row0 col4" >Pave</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col5" class="data row0 col5" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col6" class="data row0 col6" >Reg</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col7" class="data row0 col7" >Lvl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col8" class="data row0 col8" >AllPub</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col9" class="data row0 col9" >Inside</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col10" class="data row0 col10" >Gtl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col11" class="data row0 col11" >CollgCr</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col12" class="data row0 col12" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col13" class="data row0 col13" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col14" class="data row0 col14" >1Fam</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col15" class="data row0 col15" >2Story</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col16" class="data row0 col16" >7</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col17" class="data row0 col17" >5</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col18" class="data row0 col18" >2003</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col19" class="data row0 col19" >2003</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col20" class="data row0 col20" >Gable</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col21" class="data row0 col21" >CompShg</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col22" class="data row0 col22" >VinylSd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col23" class="data row0 col23" >VinylSd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col24" class="data row0 col24" >BrkFace</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col25" class="data row0 col25" >196.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col26" class="data row0 col26" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col27" class="data row0 col27" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col28" class="data row0 col28" >PConc</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col29" class="data row0 col29" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col30" class="data row0 col30" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col31" class="data row0 col31" >No</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col32" class="data row0 col32" >GLQ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col33" class="data row0 col33" >706</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col34" class="data row0 col34" >Unf</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col35" class="data row0 col35" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col36" class="data row0 col36" >150</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col37" class="data row0 col37" >856</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col38" class="data row0 col38" >GasA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col39" class="data row0 col39" >Ex</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col40" class="data row0 col40" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col41" class="data row0 col41" >SBrkr</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col42" class="data row0 col42" >856</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col43" class="data row0 col43" >854</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col44" class="data row0 col44" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col45" class="data row0 col45" >1710</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col46" class="data row0 col46" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col47" class="data row0 col47" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col48" class="data row0 col48" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col49" class="data row0 col49" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col50" class="data row0 col50" >3</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col51" class="data row0 col51" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col52" class="data row0 col52" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col53" class="data row0 col53" >8</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col54" class="data row0 col54" >Typ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col55" class="data row0 col55" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col56" class="data row0 col56" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col57" class="data row0 col57" >Attchd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col58" class="data row0 col58" >2003.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col59" class="data row0 col59" >RFn</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col60" class="data row0 col60" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col61" class="data row0 col61" >548</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col62" class="data row0 col62" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col63" class="data row0 col63" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col64" class="data row0 col64" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col65" class="data row0 col65" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col66" class="data row0 col66" >61</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col67" class="data row0 col67" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col68" class="data row0 col68" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col69" class="data row0 col69" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col70" class="data row0 col70" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col71" class="data row0 col71" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col72" class="data row0 col72" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col73" class="data row0 col73" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col74" class="data row0 col74" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col75" class="data row0 col75" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col76" class="data row0 col76" >2008</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col77" class="data row0 col77" >WD</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col78" class="data row0 col78" >Normal</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row0_col79" class="data row0 col79" >208500</td>
            </tr>
            <tr>
                        <th id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5level0_row1" class="row_heading level0 row1" >2</th>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col0" class="data row1 col0" >20</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col1" class="data row1 col1" >RL</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col2" class="data row1 col2" >80.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col3" class="data row1 col3" >9600</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col4" class="data row1 col4" >Pave</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col5" class="data row1 col5" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col6" class="data row1 col6" >Reg</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col7" class="data row1 col7" >Lvl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col8" class="data row1 col8" >AllPub</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col9" class="data row1 col9" >FR2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col10" class="data row1 col10" >Gtl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col11" class="data row1 col11" >Veenker</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col12" class="data row1 col12" >Feedr</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col13" class="data row1 col13" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col14" class="data row1 col14" >1Fam</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col15" class="data row1 col15" >1Story</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col16" class="data row1 col16" >6</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col17" class="data row1 col17" >8</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col18" class="data row1 col18" >1976</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col19" class="data row1 col19" >1976</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col20" class="data row1 col20" >Gable</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col21" class="data row1 col21" >CompShg</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col22" class="data row1 col22" >MetalSd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col23" class="data row1 col23" >MetalSd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col24" class="data row1 col24" >None</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col25" class="data row1 col25" >0.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col26" class="data row1 col26" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col27" class="data row1 col27" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col28" class="data row1 col28" >CBlock</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col29" class="data row1 col29" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col30" class="data row1 col30" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col31" class="data row1 col31" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col32" class="data row1 col32" >ALQ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col33" class="data row1 col33" >978</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col34" class="data row1 col34" >Unf</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col35" class="data row1 col35" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col36" class="data row1 col36" >284</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col37" class="data row1 col37" >1262</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col38" class="data row1 col38" >GasA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col39" class="data row1 col39" >Ex</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col40" class="data row1 col40" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col41" class="data row1 col41" >SBrkr</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col42" class="data row1 col42" >1262</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col43" class="data row1 col43" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col44" class="data row1 col44" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col45" class="data row1 col45" >1262</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col46" class="data row1 col46" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col47" class="data row1 col47" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col48" class="data row1 col48" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col49" class="data row1 col49" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col50" class="data row1 col50" >3</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col51" class="data row1 col51" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col52" class="data row1 col52" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col53" class="data row1 col53" >6</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col54" class="data row1 col54" >Typ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col55" class="data row1 col55" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col56" class="data row1 col56" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col57" class="data row1 col57" >Attchd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col58" class="data row1 col58" >1976.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col59" class="data row1 col59" >RFn</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col60" class="data row1 col60" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col61" class="data row1 col61" >460</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col62" class="data row1 col62" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col63" class="data row1 col63" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col64" class="data row1 col64" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col65" class="data row1 col65" >298</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col66" class="data row1 col66" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col67" class="data row1 col67" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col68" class="data row1 col68" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col69" class="data row1 col69" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col70" class="data row1 col70" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col71" class="data row1 col71" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col72" class="data row1 col72" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col73" class="data row1 col73" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col74" class="data row1 col74" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col75" class="data row1 col75" >5</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col76" class="data row1 col76" >2007</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col77" class="data row1 col77" >WD</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col78" class="data row1 col78" >Normal</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row1_col79" class="data row1 col79" >181500</td>
            </tr>
            <tr>
                        <th id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5level0_row2" class="row_heading level0 row2" >3</th>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col0" class="data row2 col0" >60</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col1" class="data row2 col1" >RL</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col2" class="data row2 col2" >68.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col3" class="data row2 col3" >11250</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col4" class="data row2 col4" >Pave</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col5" class="data row2 col5" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col6" class="data row2 col6" >IR1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col7" class="data row2 col7" >Lvl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col8" class="data row2 col8" >AllPub</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col9" class="data row2 col9" >Inside</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col10" class="data row2 col10" >Gtl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col11" class="data row2 col11" >CollgCr</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col12" class="data row2 col12" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col13" class="data row2 col13" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col14" class="data row2 col14" >1Fam</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col15" class="data row2 col15" >2Story</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col16" class="data row2 col16" >7</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col17" class="data row2 col17" >5</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col18" class="data row2 col18" >2001</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col19" class="data row2 col19" >2002</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col20" class="data row2 col20" >Gable</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col21" class="data row2 col21" >CompShg</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col22" class="data row2 col22" >VinylSd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col23" class="data row2 col23" >VinylSd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col24" class="data row2 col24" >BrkFace</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col25" class="data row2 col25" >162.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col26" class="data row2 col26" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col27" class="data row2 col27" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col28" class="data row2 col28" >PConc</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col29" class="data row2 col29" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col30" class="data row2 col30" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col31" class="data row2 col31" >Mn</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col32" class="data row2 col32" >GLQ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col33" class="data row2 col33" >486</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col34" class="data row2 col34" >Unf</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col35" class="data row2 col35" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col36" class="data row2 col36" >434</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col37" class="data row2 col37" >920</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col38" class="data row2 col38" >GasA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col39" class="data row2 col39" >Ex</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col40" class="data row2 col40" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col41" class="data row2 col41" >SBrkr</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col42" class="data row2 col42" >920</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col43" class="data row2 col43" >866</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col44" class="data row2 col44" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col45" class="data row2 col45" >1786</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col46" class="data row2 col46" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col47" class="data row2 col47" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col48" class="data row2 col48" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col49" class="data row2 col49" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col50" class="data row2 col50" >3</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col51" class="data row2 col51" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col52" class="data row2 col52" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col53" class="data row2 col53" >6</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col54" class="data row2 col54" >Typ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col55" class="data row2 col55" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col56" class="data row2 col56" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col57" class="data row2 col57" >Attchd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col58" class="data row2 col58" >2001.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col59" class="data row2 col59" >RFn</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col60" class="data row2 col60" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col61" class="data row2 col61" >608</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col62" class="data row2 col62" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col63" class="data row2 col63" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col64" class="data row2 col64" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col65" class="data row2 col65" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col66" class="data row2 col66" >42</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col67" class="data row2 col67" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col68" class="data row2 col68" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col69" class="data row2 col69" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col70" class="data row2 col70" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col71" class="data row2 col71" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col72" class="data row2 col72" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col73" class="data row2 col73" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col74" class="data row2 col74" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col75" class="data row2 col75" >9</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col76" class="data row2 col76" >2008</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col77" class="data row2 col77" >WD</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col78" class="data row2 col78" >Normal</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row2_col79" class="data row2 col79" >223500</td>
            </tr>
            <tr>
                        <th id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5level0_row3" class="row_heading level0 row3" >4</th>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col0" class="data row3 col0" >70</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col1" class="data row3 col1" >RL</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col2" class="data row3 col2" >60.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col3" class="data row3 col3" >9550</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col4" class="data row3 col4" >Pave</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col5" class="data row3 col5" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col6" class="data row3 col6" >IR1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col7" class="data row3 col7" >Lvl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col8" class="data row3 col8" >AllPub</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col9" class="data row3 col9" >Corner</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col10" class="data row3 col10" >Gtl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col11" class="data row3 col11" >Crawfor</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col12" class="data row3 col12" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col13" class="data row3 col13" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col14" class="data row3 col14" >1Fam</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col15" class="data row3 col15" >2Story</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col16" class="data row3 col16" >7</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col17" class="data row3 col17" >5</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col18" class="data row3 col18" >1915</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col19" class="data row3 col19" >1970</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col20" class="data row3 col20" >Gable</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col21" class="data row3 col21" >CompShg</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col22" class="data row3 col22" >Wd Sdng</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col23" class="data row3 col23" >Wd Shng</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col24" class="data row3 col24" >None</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col25" class="data row3 col25" >0.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col26" class="data row3 col26" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col27" class="data row3 col27" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col28" class="data row3 col28" >BrkTil</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col29" class="data row3 col29" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col30" class="data row3 col30" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col31" class="data row3 col31" >No</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col32" class="data row3 col32" >ALQ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col33" class="data row3 col33" >216</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col34" class="data row3 col34" >Unf</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col35" class="data row3 col35" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col36" class="data row3 col36" >540</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col37" class="data row3 col37" >756</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col38" class="data row3 col38" >GasA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col39" class="data row3 col39" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col40" class="data row3 col40" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col41" class="data row3 col41" >SBrkr</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col42" class="data row3 col42" >961</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col43" class="data row3 col43" >756</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col44" class="data row3 col44" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col45" class="data row3 col45" >1717</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col46" class="data row3 col46" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col47" class="data row3 col47" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col48" class="data row3 col48" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col49" class="data row3 col49" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col50" class="data row3 col50" >3</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col51" class="data row3 col51" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col52" class="data row3 col52" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col53" class="data row3 col53" >7</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col54" class="data row3 col54" >Typ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col55" class="data row3 col55" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col56" class="data row3 col56" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col57" class="data row3 col57" >Detchd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col58" class="data row3 col58" >1998.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col59" class="data row3 col59" >Unf</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col60" class="data row3 col60" >3</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col61" class="data row3 col61" >642</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col62" class="data row3 col62" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col63" class="data row3 col63" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col64" class="data row3 col64" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col65" class="data row3 col65" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col66" class="data row3 col66" >35</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col67" class="data row3 col67" >272</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col68" class="data row3 col68" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col69" class="data row3 col69" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col70" class="data row3 col70" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col71" class="data row3 col71" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col72" class="data row3 col72" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col73" class="data row3 col73" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col74" class="data row3 col74" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col75" class="data row3 col75" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col76" class="data row3 col76" >2006</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col77" class="data row3 col77" >WD</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col78" class="data row3 col78" >Abnorml</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row3_col79" class="data row3 col79" >140000</td>
            </tr>
            <tr>
                        <th id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5level0_row4" class="row_heading level0 row4" >5</th>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col0" class="data row4 col0" >60</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col1" class="data row4 col1" >RL</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col2" class="data row4 col2" >84.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col3" class="data row4 col3" >14260</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col4" class="data row4 col4" >Pave</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col5" class="data row4 col5" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col6" class="data row4 col6" >IR1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col7" class="data row4 col7" >Lvl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col8" class="data row4 col8" >AllPub</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col9" class="data row4 col9" >FR2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col10" class="data row4 col10" >Gtl</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col11" class="data row4 col11" >NoRidge</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col12" class="data row4 col12" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col13" class="data row4 col13" >Norm</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col14" class="data row4 col14" >1Fam</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col15" class="data row4 col15" >2Story</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col16" class="data row4 col16" >8</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col17" class="data row4 col17" >5</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col18" class="data row4 col18" >2000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col19" class="data row4 col19" >2000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col20" class="data row4 col20" >Gable</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col21" class="data row4 col21" >CompShg</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col22" class="data row4 col22" >VinylSd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col23" class="data row4 col23" >VinylSd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col24" class="data row4 col24" >BrkFace</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col25" class="data row4 col25" >350.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col26" class="data row4 col26" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col27" class="data row4 col27" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col28" class="data row4 col28" >PConc</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col29" class="data row4 col29" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col30" class="data row4 col30" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col31" class="data row4 col31" >Av</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col32" class="data row4 col32" >GLQ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col33" class="data row4 col33" >655</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col34" class="data row4 col34" >Unf</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col35" class="data row4 col35" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col36" class="data row4 col36" >490</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col37" class="data row4 col37" >1145</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col38" class="data row4 col38" >GasA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col39" class="data row4 col39" >Ex</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col40" class="data row4 col40" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col41" class="data row4 col41" >SBrkr</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col42" class="data row4 col42" >1145</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col43" class="data row4 col43" >1053</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col44" class="data row4 col44" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col45" class="data row4 col45" >2198</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col46" class="data row4 col46" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col47" class="data row4 col47" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col48" class="data row4 col48" >2</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col49" class="data row4 col49" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col50" class="data row4 col50" >4</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col51" class="data row4 col51" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col52" class="data row4 col52" >Gd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col53" class="data row4 col53" >9</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col54" class="data row4 col54" >Typ</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col55" class="data row4 col55" >1</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col56" class="data row4 col56" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col57" class="data row4 col57" >Attchd</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col58" class="data row4 col58" >2000.000000</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col59" class="data row4 col59" >RFn</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col60" class="data row4 col60" >3</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col61" class="data row4 col61" >836</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col62" class="data row4 col62" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col63" class="data row4 col63" >TA</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col64" class="data row4 col64" >Y</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col65" class="data row4 col65" >192</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col66" class="data row4 col66" >84</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col67" class="data row4 col67" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col68" class="data row4 col68" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col69" class="data row4 col69" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col70" class="data row4 col70" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col71" class="data row4 col71" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col72" class="data row4 col72" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col73" class="data row4 col73" >nan</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col74" class="data row4 col74" >0</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col75" class="data row4 col75" >12</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col76" class="data row4 col76" >2008</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col77" class="data row4 col77" >WD</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col78" class="data row4 col78" >Normal</td>
                        <td id="T_c8024f74_6abf_11eb_8180_a45e60b8a5e5row4_col79" class="data row4 col79" >250000</td>
            </tr>
    </tbody></table>




```python
from IPython.display import display
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
