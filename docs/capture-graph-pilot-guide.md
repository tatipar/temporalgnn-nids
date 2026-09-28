# Guía operativa para el piloto de grafos en cAPTure

## 1. Propósito y alcance

Este documento convierte la recomendación de hacer una prueba preliminar de
grafos en un protocolo ejecutable y auditable. El piloto debe responder una
pregunta acotada:

> ¿Los cinco escenarios de desarrollo contienen variación relacional que una
> GNN pueda aprovechar y esa información mejora la detección temprana a un
> costo operativo razonable?

El piloto es **exploratorio y de desarrollo**. Sirve para decidir si la
topología merece ocupar un lugar central en la propuesta; no confirma la
superioridad de una arquitectura y no autoriza conclusiones sobre
generalización final.

Queda fuera de alcance:

- acceder a Test1 o Test2;
- inspeccionar `train_pub_exf` o `train_user_prop`;
- cambiar la ventana primaria usando resultados de modelos;
- seleccionar el mejor seed;
- afirmar que una diferencia observada con un único seed es estable;
- interpretar una mejora agregada como evidencia topológica sin los controles
  de ablación.

Esta guía complementa, sin reemplazar, el
[plan experimental de cAPTure](capture-experimental-plan.md), el
[manifiesto versionado](../configs/capture_experiment_v1.yaml) y el
[contrato de estado temporal y ablaciones](temporal-state-and-ablation-contract.md).
En caso de discrepancia, primero se corrige y versiona el manifiesto; no se
resuelve la diferencia con un parámetro implícito en un notebook.

La Etapa 1 se ejecuta con el notebook
[`capture_graph_structural_audit.ipynb`](../code/python/notebook/capture_graph_structural_audit.ipynb)
y el contrato
[`capture_graph_structural_audit_v1.yaml`](../configs/capture_graph_structural_audit_v1.yaml).

## 2. Contrato congelado para todo el piloto

| Elemento | Decisión |
|---|---|
| Datos | Sólo los cinco escenarios de desarrollo |
| Escenarios | `train_empty_conn`, `train_qos_mid`, `train_dollar_char`, `train_slash_char`, `train_sub_exf` |
| Fold A | entrena con `empty_conn`, `qos_mid`; evalúa en `dollar_char`, `slash_char`, `sub_exf` |
| Fold B | entrena con `dollar_char`, `slash_char`, `sub_exf`; evalúa en `empty_conn`, `qos_mid` |
| Seed | `42` |
| Unidad de predicción | un paquete, representado por una arista |
| Ventanas | fijas, no solapadas, semiabiertas y de 5 segundos |
| Origen primario | primer timestamp del escenario |
| Tiempo de decisión | `decision_time = window_end` para todos los modelos |
| Grafo | multigrafo dirigido; se conservan aristas paralelas |
| Nodos | MAC Ethernet normalizada; identidades usadas sólo para construir topología |
| Features | mismo vector de paquete preprocesado dentro de cada fold |
| Preprocesamiento | ajustado únicamente con los escenarios de entrenamiento del fold |
| Pesos de entrenamiento | política escenario/clase ya declarada, ajustada sólo en el entrenamiento del fold |
| Estado temporal | orden cronológico, sin shuffle; reset en escenario, época, pasada de evaluación y fold |
| Alerta de ventana | máximo score de paquete de la ventana mayor o igual al umbral |
| Presupuesto primario | una ventana de falsa alerta por hora |
| Agregación | escenarios dentro de fold y luego folds; no usar sólo micro-promedios de paquetes |

Las ventanas vacías no se materializan como grafos, pero sí cuentan como
exposición temporal en el denominador de falsas alertas. Los saltos de índice
de ventana deben conservarse para que el modelo temporal conozca el tiempo
transcurrido.

## 3. Decisiones que deben cerrarse antes de entrenar

El manifiesto todavía declara como no resueltos la política de memoria y los
tamaños mínimos de efecto. Antes de observar resultados del piloto se debe
crear una revisión del manifiesto que congele:

- arquitectura exacta, dimensión oculta, dropout, optimizador, learning rate y
  weight decay de cada modelo;
- política de memoria, semivida o corte de gaps;
- número máximo de épocas, paciencia y mejora mínima del early stopping;
- regla del subconjunto cronológico interno usado para checkpointing;
- métrica de checkpoint, independiente de la selección operativa del modelo;
- cantidad máxima de parámetros o la regla para igualar capacidad;
- mejoras mínimas que se considerarán materialmente relevantes;
- costo máximo aceptable de construcción, inferencia y memoria;
- tolerancia al cambio de origen de ventana.

No se debe elegir esos valores después de comparar scores. Si se usa como
punto de partida la configuración MLP ya congelada, la decisión debe quedar
registrada como reutilización deliberada y no como una búsqueda de
hiperparámetros para la GNN.

### Early stopping sin contaminar el fold externo

El fold externo produce las predicciones OOF que se reportarán. Por ello, no
debe seleccionar la época ni el checkpoint.

Para cada fold y escenario de entrenamiento:

1. ordenar ventanas cronológicamente;
2. separar un tramo final interno, en límites completos de ventana, sólo para
   checkpointing;
3. entrenar con el tramo anterior;
4. elegir la época mediante la métrica interna congelada;
5. restaurar ese checkpoint una sola vez;
6. evaluar los escenarios externos sin volver a ajustar pesos, época o
   umbral.

La fracción y la métrica internas deben fijarse antes del primer entrenamiento.
Si no puede definirse un tramo interno con clases y steps adecuados, usar una
cantidad fija de épocas común a todos los modelos es más limpio que hacer
early stopping sobre el fold externo. La separación interna es una herramienta
de checkpointing, no una tercera estimación reportable de generalización.

## 4. Etapa 1: auditoría estructural sin entrenamiento

### 4.1 Construcción auditable

Construir los grafos escenario por escenario, con la misma función que luego
consumirán los modelos. Para cada paquete debe poder verificarse:

```text
packet_id -> scenario -> window_id -> edge_index -> edge_attr -> y
```

Controles obligatorios:

- una arista y un target por cada paquete canónico;
- ningún paquete duplicado, perdido o asignado a dos ventanas;
- orden de aristas reconstruible mediante `packet_id` o `source_row_id`;
- endpoints no nulos y normalizados según el esquema;
- aristas paralelas conservadas;
- timestamps de grafo estrictamente crecientes dentro de cada escenario;
- ningún grafo ni estado compartido entre escenarios o folds;
- igualdad exacta entre labels del paquete y de su arista;
- hash de los paquetes preparados, preprocesador, configuración y artefactos
  construidos.

Primero ejecutar un smoke con un escenario de cada bloque benigno. Sólo después
de revisar sus invariantes construir los cinco escenarios.

### 4.2 Tabla por ventana

Guardar una fila por ventana no vacía, más un resumen de ventanas vacías, con
al menos estas columnas:

| Grupo | Campos mínimos |
|---|---|
| Identidad | escenario, bloque benigno, índice, inicio, fin, duración |
| Tamaño | paquetes/aristas, nodos, pares dirigidos únicos, pares no dirigidos únicos |
| Conectividad | componentes débiles, fracción de nodos en la mayor componente |
| Carga relacional | densidad simple dirigida, multiplicidad media y máxima, fracción de aristas paralelas |
| Cambio | Jaccard de nodos y pares respecto de la ventana anterior, nodos nuevos, nodos retenidos |
| Labels | benignos, ataques, tipo de ventana, steps presentes, iteraciones presentes |
| Protocolo | conteos por capa y fracciones de broadcast/multicast, ARP, IPv4/IPv6, TCP/UDP, MQTT/SSH |
| Recursos | tiempo de construcción, bytes serializados y pico de RAM atribuible |

Definiciones recomendadas, con `m` aristas, `n` nodos y `q` pares dirigidos
únicos sin contar multiplicidad:

```text
density_simple = q / (n * (n - 1))          si n > 1 y no se cuentan self-loops
mean_multiplicity = m / q                    si q > 0
parallel_edge_fraction = (m - q) / m         si m > 0
jaccard_nodes(t) = |V_t inter V_t-1| / |V_t union V_t-1|
jaccard_pairs(t) = |P_t inter P_t-1| / |P_t union P_t-1|
```

La densidad de un multigrafo no debe calcularse directamente con `m`, porque
puede superar uno y confunde conectividad con volumen. Reportar por separado
la densidad del grafo simple y la multiplicidad. Declarar cómo se tratan los
self-loops aunque no aparezcan.

Calcular cambio de endpoints de dos maneras:

- entre índices de ventana adyacentes, considerando explícitamente los gaps;
- entre grafos no vacíos consecutivos, registrando cuántas ventanas vacías los
  separan.

Así no se interpreta como continuidad una comparación que saltó minutos de
tiempo real.

### 4.3 Resúmenes requeridos

Para cada escenario y para el total jerárquico, producir:

- cantidad de ventanas totales, vacías y no vacías;
- percentiles 0, 1, 5, 25, 50, 75, 95, 99 y 100 de nodos, aristas, pares,
  componentes, multiplicidad y tamaño serializado;
- proporción de ventanas benignas puras, mixtas y de ataque puro;
- duración continua cubierta y densidad de ventanas ocupadas;
- distribución de nodos y pares nuevos/retenidos;
- frecuencia de cada conjunto de endpoints o pares repetido exactamente;
- topologías más frecuentes y porcentaje de ventanas explicado por ellas;
- métricas topológicas separadas por benigno puro, mixto, ataque puro y por
  step, siempre con cantidad de ventanas junto al resumen;
- tiempo total, tiempo por millón de paquetes, almacenamiento y pico de RAM.

Las comparaciones por step son descriptivas. Deben acompañarse por escenario y
tipo de ventana para no confundir topología de ataque con volumen, protocolo o
bloque benigno. Si se calculan diferencias estandarizadas o intervalos por
bootstrap, el remuestreo debe respetar escenario e iteración; las ventanas no
son observaciones independientes.

### 4.4 Auditoría de atajos triviales

Esta parte no entrena la GNN. Busca señales que podrían explicar un resultado
sin aprendizaje relacional generalizable:

- prevalencia de ataque por protocolo y por combinaciones simples de
  indicadores;
- prevalencia por tipo de dirección MAC: unicast, multicast, broadcast y nodo
  de grupo;
- porcentaje de paquetes de ataque identificable por una única regla de
  protocolo o rol de puerto;
- endpoints y pares exclusivos de ataque dentro de cada escenario;
- cobertura en el fold externo de endpoints o pares vistos como ataque en el
  entrenamiento del fold;
- rendimiento de reglas de memorización de endpoint/par, usando únicamente el
  entrenamiento del fold para construir la regla;
- resultados estratificados MQTT/no MQTT y por protocolo dominante;
- repetición exacta de fondos benignos o ventanas estructuralmente idénticas.

Las MAC crudas, OUI, IDs globales, nombres de escenario y steps nunca se
incorporan a `edge_attr`. Pueden utilizarse en esta auditoría como metadatos
diagnósticos, siempre que el reporte diferencie con claridad una fuga o atajo
de una feature permitida.

### 4.5 Salidas de la etapa 1

La etapa queda completa cuando existen:

```text
graph_pilot/<run_id>/
  resolved_config.yaml
  provenance.json
  construction_summary.json
  window_metrics.parquet
  scenario_summary.csv
  percentile_summary.csv
  step_topology_summary.csv
  shortcut_audit.json
  resource_summary.json
  figures/
  review.md
```

`review.md` debe contestar, con evidencia:

1. ¿Cambian los nodos y enlaces entre ventanas?
2. ¿La mayor parte del tráfico es un conjunto pequeño de pares repetidos?
3. ¿Hay estructura conectada suficiente para propagación de mensajes?
4. ¿Las diferencias por step sobreviven al desglose por escenario y protocolo?
5. ¿La construcción completa cabe en el entorno previsto?
6. ¿Un atajo de MAC o protocolo podría dominar cualquier aparente ganancia?

### 4.6 Gate estructural

Continuar a la etapa 2 sólo si la construcción es íntegra y cabe en recursos.
Una topología casi constante, componentes triviales de dos nodos, ausencia de
vecindarios compartidos o una regla simple que explica casi todos los ataques
son razones para **depriorizar** la GNN. No son una razón para descartar memoria
temporal, el MLP o los resúmenes causales ya existentes.

La variación estructural es una condición favorable, no prueba que sea
predictiva. La decisión final sobre utilidad topológica necesita las
ablaciones de la etapa 2.

## 5. Etapa 2: comparación de un seed en desarrollo

### 5.1 Matriz mínima

Todos los modelos reciben exactamente los mismos paquetes, features, folds,
pesos de entrenamiento, ventanas y tiempos de decisión.

| Variante | Clase conceptual | Información disponible | Pregunta |
|---|---|---|---|
| Edge MLP | `SimpleMLP` | paquete actual | ¿Qué logra una red sin tiempo ni relaciones? |
| EdgeGRU | `EdgeGRU_Baseline_NoX` | paquete + memoria por nodo | ¿Agrega valor la memoria sin GAT? |
| StaticGNN | `StaticGNN_Identity` | paquete + topología actual | ¿Agrega valor la propagación actual sin memoria recurrente? |
| ST-GNN | `ST_GNN_Identity` | paquete + topología + memoria | ¿Se complementan estructura y tiempo? |
| ST-GNN sin GAT | `ST_GNN_Identity(use_topology=false)` | paquete + agregación por endpoint + memoria, sin message passing GAT | Control principal de las capas topológicas |
| ST-GNN sin acceso directo a `edge_attr` | `ST_GNN_Identity(use_direct_edge_attr=false)` | edge features sólo a través de identidad/agregación y GAT | ¿El clasificador depende del atajo directo del paquete? |

La última variante es condicional al costo y se ejecuta sólo si se declaró
antes de ver los resultados de la matriz mínima.

**Precisión conceptual importante:** en la implementación actual,
`use_topology=false` evita las capas GATv2, pero conserva agregados locales de
aristas entrantes y salientes para formar la identidad del nodo. Por eso debe
llamarse “sin GAT” o “sin message passing”, no “sin toda topología”. Asimismo,
`use_direct_edge_attr=false` no elimina las features del paquete de la
construcción de identidad ni de los mensajes GAT. Edge MLP y EdgeGRU son los
controles sin propagación de mensajes necesarios para completar la
interpretación.

### 5.2 Igualdad experimental

Antes de lanzar una corrida, un test de contrato debe demostrar que las
variantes tienen:

- los mismos IDs y orden de paquetes en entrenamiento y evaluación;
- el mismo `edge_attr`, target y peso por paquete;
- la misma asignación a ventanas;
- el mismo origen y `window_end`;
- el mismo fold y subconjunto interno de checkpoint;
- el mismo umbral operativo definido a partir de OOF, no uno elegido en Test;
- resets y orden temporal correctos;
- parámetros y FLOPs reportados, aunque no sean idénticos.

No añadir resúmenes XGB-P+T a una variante aislada. Si se desea una segunda
matriz con los seis features históricos causales, debe darse el mismo vector a
todas las variantes y etiquetarse como un experimento separado.

### 5.3 Entrenamiento y persistencia

Para cada combinación de fold y modelo:

1. fijar seed `42` para Python, NumPy, PyTorch y CUDA;
2. cargar sólo los escenarios de entrenamiento del fold;
3. ajustar preprocessing y pesos sólo con ese conjunto;
4. entrenar en orden cronológico cuando exista estado temporal;
5. hacer early stopping sólo con el tramo interno predeclarado;
6. restaurar el mejor checkpoint interno;
7. resetear toda memoria;
8. inferir una única vez sobre cada escenario externo;
9. persistir un score por paquete, junto con claves temporales y de evaluación;
10. registrar duración, pico de RAM/VRAM, parámetros y tamaño del checkpoint.

Los modelos no temporales pueden barajar ejemplos para optimización sólo si no
cambia el conjunto ni los pesos. Para comparabilidad de inferencia, todos se
evalúan como secuencias de ventanas en el mismo orden.

El tiempo de inferencia debe medirse sin carga de archivos y también extremo a
extremo. Reportar, como mínimo, segundos por millón de paquetes y percentiles
de latencia por ventana después de un warm-up explícito. Sincronizar CUDA antes
y después de cada medición.

### 5.4 Umbral y semántica operativa

Una ventana alerta cuando:

```text
max(packet_score en la ventana) >= threshold
```

Para cada modelo, el umbral de desarrollo se obtiene de sus predicciones OOF:
el menor umbral cuyo peor promedio de fold de falsas alertas por escenario
cumple una ventana falsa por hora. Deben conservarse las reglas ya declaradas
para empates con `nextafter` y comparación en `float64`.

Esta calibración y su evaluación usan las mismas predicciones OOF, por lo que
los resultados operativos del piloto pueden ser optimistas. Son adecuados para
screening, no para una cifra confirmatoria.

## 6. Métricas obligatorias

### 6.1 Detección y operación

Reportar por modelo, fold, escenario y step:

- ROC-AUC y PR-AUC de paquete como diagnósticos;
- precisión, recall y FPR de paquete al umbral operativo;
- ventanas de falsa alerta por hora;
- iteraciones totales, detectadas y perdidas;
- cobertura de iteraciones con una alerta correcta estrictamente anterior al
  último paquete malicioso de la iteración;
- cobertura antes de la acción terminal declarada;
- cobertura por tipo de step;
- primer tiempo de alerta y lead time;
- cantidad de escenarios y steps donde cambia el signo de la diferencia frente
  al control.

Definiciones temporales:

```text
alert_time = window_end de la primera ventana con un paquete malicioso que cruza el umbral
timely_step = alert_time < timestamp del último paquete malicioso de la iteración
early_terminal = alert_time < inicio de la primera acción terminal declarada
terminal_lead_time = terminal_onset - alert_time
```

Los misses permanecen como misses; no se convierten en una detección al final
del step. El lead time se resume únicamente junto con la cobertura y la
cantidad de misses. Informar mediana y percentiles además de la media, porque
la ventana de 5 segundos cuantiza el tiempo.

### 6.2 Costo

Separar siempre:

- tiempo único de construcción de grafos;
- tiempo de entrenamiento hasta el checkpoint elegido;
- tiempo de inferencia puro;
- tiempo de inferencia extremo a extremo;
- pico de RAM y VRAM;
- tamaño de grafos y checkpoints en disco;
- cantidad de parámetros y ventanas/paquetes procesados por segundo.

### 6.3 Sensibilidad al origen

Después de cerrar la matriz primaria, reconstruir ventanas con un desplazamiento
de `+2.5 s`. La sensibilidad mínima incluye Edge MLP y las variantes que
sustenten una conclusión topológica; no es necesario repetir automáticamente
toda la escalera si el costo es prohibitivo.

No reoptimizar arquitectura ni ancho de ventana. Aplicar el protocolo
predeclarado y reportar:

- cambio absoluto de cobertura oportuna;
- cambio de falsas alertas por hora;
- cambio de lead time;
- cambio de tamaño y costo de grafos;
- si se conserva el signo de las comparaciones topológicas principales.

El origen desplazado es una sensibilidad de desarrollo, no una oportunidad
para reemplazar retrospectivamente el origen primario.

## 7. Lectura causal de las comparaciones

| Comparación | Evidencia principal | Limitación |
|---|---|---|
| EdgeGRU - Edge MLP | valor de memoria por endpoint | también cambia la arquitectura |
| StaticGNN - Edge MLP | valor de agregación/message passing actual | puede cambiar capacidad y optimización |
| ST-GNN - StaticGNN | valor incremental de memoria con estructura | requiere configuraciones alineadas |
| ST-GNN - EdgeGRU | valor incremental de GAT en un modelo temporal | no aísla perfectamente interacciones |
| ST-GNN - ST-GNN sin GAT | valor de las capas GATv2 dentro de la misma familia | el control aún usa agregación por endpoint |
| ST-GNN sin `edge_attr` directo - ST-GNN | dependencia del atajo directo al clasificador | las features siguen presentes en identidad y mensajes |

Una mejora sólo en AUC no demuestra utilidad operativa. La evidencia relevante
es una mejora consistente de cobertura o lead time al mismo presupuesto de
falsas alertas, acompañada por desglose de escenario/step y costo.

## 8. Regla de decisión del piloto

Antes de entrenar, completar en el manifiesto los campos entre corchetes:

```text
mejora_mínima_cobertura_oportuna = [valor]
mejora_mínima_cobertura_terminal = [valor]
mejora_mínima_lead_time = [valor y estadístico]
escenarios_con_signo_positivo_mínimos = [valor de 5]
degradación_máxima_por_origen_desplazado = [valor]
costo_máximo_inferencia = [valor]
pico_máximo_ram_vram = [valor]
```

### La topología merece avanzar

Sólo si se cumplen conjuntamente:

- StaticGNN o ST-GNN supera su control no topológico en una métrica operativa
  predeclarada al mismo presupuesto de falsas alertas;
- la diferencia alcanza el tamaño mínimo práctico congelado;
- el signo no depende de un único escenario, step o protocolo;
- la conclusión principal se conserva con origen desplazado;
- el resultado no se explica por identidades MAC o una regla trivial de
  protocolo;
- construcción e inferencia cumplen el presupuesto de recursos.

El siguiente paso sería una confirmación multi-seed en desarrollo antes de
congelar el pipeline final. Test1 y Test2 continúan cerrados.

### La evidencia favorece tiempo pero no topología

Si EdgeGRU mejora al Edge MLP, pero StaticGNN/ST-GNN no mejoran a los controles
sin GAT, priorizar memoria temporal o el baseline `history` ya disponible. La
GNN puede quedar como análisis secundario, no como contribución central.

### La topología se deprioriza

Si la auditoría muestra grafos casi constantes o triviales y las ablaciones no
aportan mejoras operativas robustas, documentar el resultado negativo y evitar
una búsqueda amplia de GNN. Esto es una conclusión válida del piloto, no un
fallo de ejecución.

### Resultado inconcluso

Marcarlo como inconcluso si hay fallos de contrato, memoria insuficiente,
inestabilidad severa con el origen, dependencia de un único escenario o
diferencias menores que los mínimos predeclarados. No resolverlo mirando Test.

## 9. Entregables de la etapa 2

```text
graph_pilot/<run_id>/
  resolved_config.yaml
  provenance.json
  graph_contract_report.json
  models/<model>/<fold>/
    checkpoint.pt
    training_history.json
    timing.json
    resource_usage.json
    predictions.parquet
  threshold_selection.json
  packet_metrics.csv
  window_metrics.csv
  iteration_metrics.csv
  step_metrics.csv
  scenario_metrics.csv
  origin_shift_metrics.csv
  comparison_table.csv
  figures/
  pilot_decision.md
```

Cada fila de `predictions.parquet` debe conservar al menos:

```text
model, fold, scenario, packet_id, source_row_id, window_id,
window_start, window_end, packet_timestamp, binary_label,
attack_step, sequence_id, score
```

`attack_step` y `sequence_id` son metadatos exclusivos de evaluación. Nunca se
entregan al modelo.

## 10. Checklist de ejecución

### Antes de construir

- [ ] Test1, Test2 y escenarios prohibidos siguen inaccesibles.
- [ ] Manifest, schemas y artefactos preparados tienen hashes verificados.
- [ ] Política de self-loops, broadcast, multicast y no-IP coincide con el manifiesto.
- [ ] Métricas estructurales y fórmulas están congeladas.
- [ ] Presupuestos de almacenamiento, RAM y tiempo están declarados.

### Antes de entrenar

- [ ] Etapa 1 revisada y gate estructural documentado.
- [ ] Configuraciones exactas y tamaños mínimos de efecto están versionados.
- [ ] Split cronológico interno o alternativa de épocas fijas está congelado.
- [ ] Test sintético verifica correspondencia paquete-arista-output.
- [ ] Test sintético verifica resets, gaps y orden temporal.
- [ ] Todas las variantes reciben la misma vista de features.
- [ ] Se verificó el significado exacto de cada ablación.

### Antes de decidir

- [ ] Hay una predicción OOF por paquete y modelo.
- [ ] Umbrales usan sólo OOF de desarrollo y la regla de una falsa alerta/hora.
- [ ] Cobertura oportuna y terminal usan desigualdad estricta.
- [ ] Misses, denominadores y tamaños de muestra están visibles.
- [ ] Resultados están separados por fold, escenario, step y protocolo.
- [ ] Se informan tiempo, RAM/VRAM y almacenamiento.
- [ ] Se ejecutó la sensibilidad de origen predeclarada.
- [ ] La decisión se tomó con los criterios congelados, no sólo con AUC.
- [ ] El reporte dice explícitamente que el piloto usa un solo seed.

## 11. Tabla compacta para `pilot_decision.md`

| Pregunta | Evidencia | Resultado | Decisión |
|---|---|---|---|
| ¿Hay variación real de nodos y enlaces? | percentiles, Jaccard, topologías repetidas |  |  |
| ¿Hay vecindarios no triviales? | componentes, mayor componente, grados, multiplicidad |  |  |
| ¿Existen atajos de MAC/protocolo? | auditoría de reglas y estratos |  |  |
| ¿La topología mejora cobertura oportuna? | StaticGNN/MLP y ST-GNN/sin GAT |  |  |
| ¿La memoria mejora cobertura o lead time? | EdgeGRU/MLP y ST-GNN/StaticGNN |  |  |
| ¿Se detecta antes de la acción terminal? | cobertura y lead time terminal |  |  |
| ¿Cumple una falsa alerta por hora? | peor promedio de fold |  |  |
| ¿Es estable al origen `+2.5 s`? | sensibilidad de ventana |  |  |
| ¿Cabe en recursos? | tiempos, throughput, RAM/VRAM, disco |  |  |
| ¿Debe avanzar a multi-seed? | criterios predeclarados completos |  |  |
