**Foco de problema:** Materias primas 

**Contexto del problema**
La materias primas tienen relevancia en el proceso de producción, dado que son importantes en el planeacion de proceso. EL area de **planeacion de la produccion** es quien se encarga de generar el estimado global de la operación. Se ha observado que hay paros de produccion debido a al escases de materia prima, esto está sustentado en base a todo el reporting de la empresa (Dashboard BI)
Dentro de producción se encuentra el area  abastecimiento  que son los encargados de conectar el area de compras y producción. Su función es la de administrar el stock según las necesidades de producción. 

**El problema**
hay paros de producción debido a la falta de materia prima en ciertos periodos de producción

**Objetivo:** Implementar una solución basada en datos que permita optimizar el control del inventario, ayudando a la empresa a una optimización de recursos de cara a la planeación 

**Fuentes de datos:**

| Tabla               | informacion                                   | Archivo |
| ------------------- | --------------------------------------------- | ------- |
| consumodeprodcutos  | transaccional de la operación de producción   | .xls    |
| inventariovsconsumo | transaccional de los consumos del inventarios | .xls    |
| tablaprovedores     | transaccional de ingresos                     | .xls    |

**Propuestas viables de predicción:** 

- Costos 
- Consumo;  Optimizar las materias primas desde la planeacion de producción, esd decir; ajustar el stock desde la sección previa a iniciar una lote  de producción, permitiendo de forma temprana tener información sobre el consumo estimado de materia prima 
- Proveedores; Modelo que permita estimar el tiempo de entrega de la materia prima una vez se realiza la compra
