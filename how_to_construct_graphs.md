# Especificación de Construcción de Grafos para NIDS Temporal con GNN + LSTM (+ Atención)

Este documento define cómo construir grafos a partir del dataset CIC-IDS2018 (NF-v3) para:

- detectar ataques de forma temprana (antes de *Impact*),
- preservar trazabilidad hacia flows y tipos de ataque,
- permitir explicabilidad (atención en nodos/edges),
- y habilitar un segundo modelo para mapear ataques → tácticas MITRE.

---

## 1. Definiciones

- **Flow**: fila del dataset (índice `flow_id`).
- **Ventana temporal** \(W_t\): intervalo \([T_t, T_t+\Delta T)\).
- **Grafo por ventana** \(G_t=(V_t,E_t)\):
  - \(V_t\): IPs activas en la ventana.
  - \(E_t\): aristas dirigidas construidas a partir de flows en la ventana.

> **Importante:**  
> una arista dirigida existe **sólo si hay flows reales en esa dirección en la ventana**  
> (NO se crea artificialmente el inverso).

---

## 2. Ventanas deslizantes

### Parámetros

- `WINDOW_SIZE = ΔT` (p. ej., 60 s / 300 s)
- `WINDOW_STEP` (puede ser < ΔT para solapamiento)

### Asignación de flows a ventanas

Usar `FLOW_START_MILLISECONDS` (o un timestamp equivalente):

1. Convertir `FLOW_START_MILLISECONDS` a tiempo relativo o datetime.
2. Un flow \(f\) pertenece a la ventana \(W_t\) si su `FLOW_START` está en \([T_t, T_t + \Delta T)\).

> Nota: para simplificar, se asigna cada flow a la ventana donde empieza. No se divide un mismo flow entre múltiples ventanas.

### Generación

- Ordenar todos los flows por tiempo de inicio.
- Definir \(T_0\) como el tiempo del primer flow (o el primer múltiplo de \(\Delta T\) que lo contenga).
- Generar ventanas \(W_t\) con:
  - \(T_t = T_0 + t \cdot \text{WINDOW_STEP}\)
  
---

## 3. Grafo por ventana

### 3.1 Nodos \(V_t\)

- Cada nodo representa una **IP**.
- Conjunto de nodos:

  \[
  V_t = \{ \text{IP} \mid \exists f \in W_t \text{ tal que } \text{IP} = \text{IPV4\_SRC\_ADDR}(f) \text{ o } \text{IPV4\_DST\_ADDR}(f) \}
  \]

Es decir, todas las IPs que participan al menos en un flow en esa ventana.

### 3.2 Aristas dirigidas \(E_t\)

Para cada par de IPs:

- Sea \(\mathcal{F}_{A\to B}^t\) el conjunto de flows con `SRC=A`, `DST=B`, en \(W_t\).
- Crear una arista dirigida \(e=(A,B)\) **sólo si** \(\mathcal{F}_{A\to B}^t\neq\emptyset\).

> Si existen flows en ambas direcciones, habrá **dos** aristas:  
> \(A\to B\) y \(B\to A\), cada una resumida **por separado**.

---

## 4. Features de aristas (edge features)

Cada arista dirigida \(e = (A,B)\) en ventana \(W_t\) resume todos los flows \(\mathcal{F}_{A\to B}^t\).

Para cada atributo del dataset, se define una función de agregación.

### 4.1. Conjunto de flows por edge

- Para cada arista dirigida \(e = (A,B)\):

  \[
  \mathcal{F}_{A\to B}^t = \{ f \in W_t \mid \text{IPV4\_SRC\_ADDR}(f) = A, \text{IPV4\_DST\_ADDR}(f) = B \}
  \]

- Definir:
  - \(n_e = |\mathcal{F}_{A\to B}^t|\) (número de flows de ese edge en la ventana).

Este `n_e` se guarda como feature `flow_count`.

### 4.2. Features numéricas acumulativas (sumas)

Para atributos que representan “cantidad” o “volumen”, se usa **suma**:

- `IN_BYTES`, `OUT_BYTES`
- `IN_PKTS`, `OUT_PKTS`
- `RETRANSMITTED_IN_BYTES`, `RETRANSMITTED_IN_PKTS`
- `RETRANSMITTED_OUT_BYTES`, `RETRANSMITTED_OUT_PKTS`

Por ejemplo, para `IN_BYTES`:

\[
\text{IN\_BYTES\_SUM}(e) = \sum_{f \in \mathcal{F}_{A\to B}^t} \text{IN\_BYTES}(f)
\]

> Recomendado: aplicar `log1p` a estas sumas antes de normalizar.

### 4.3. Features numéricas intrínsecas (mean / std / min / max)

Para atributos que describen propiedades del flow, no cantidades acumulativas:

- `FLOW_DURATION_MILLISECONDS`
- `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`
- `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`
- `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`
- `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`
- IATs:
  - `SRC_TO_DST_IAT_MIN / MAX / AVG / STDDEV`
  - `DST_TO_SRC_IAT_MIN / MAX / AVG / STDDEV`
- `MIN_TTL`, `MAX_TTL`
- `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`

Se agregan típicamente mediante:

- **media**: \(\mu\)
- **desviación estándar**: \(\sigma\)
- y opcionalmente **mínimo** / **máximo** en algunos casos.

Ejemplo para `FLOW_DURATION_MILLISECONDS`:

\[
\mu_{\text{duration}}(e) = \frac{1}{n_e} \sum_{f \in \mathcal{F}_{A\to B}^t} \text{FLOW\_DURATION\_MILLISECONDS}(f)
\]

\[
\sigma_{\text{duration}}(e) = \sqrt{\frac{1}{n_e} \sum_{f \in \mathcal{F}_{A\to B}^t} \left(\text{FLOW\_DURATION\_MILLISECONDS}(f) - \mu_{\text{duration}}(e)\right)^2}
\]

### 4.4. TCP flags y atributos pseudo-numéricos

Atributos como:

- `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`
- `ICMP_TYPE`, `ICMP_IPV4_TYPE`
- `DNS_QUERY_TYPE`

no deben tratarse como números continuos, sino como **categóricos** o conjuntos de bits.

Estrategia:

1. Transformar cada uno en una representación **one-hot** o multi-hot a nivel flow.
2. Agregar mediante:
   - **suma** → cuentas por categoría/flag.
   - y opcionalmente normalizar por \(n_e\) → frecuencia relativa.

Ejemplo:

- para `TCP_FLAGS` con codificación one-hot \(v(f)\):

\[
\text{TCP\_FLAGS\_COUNT}(e) = \sum_{f \in \mathcal{F}_{A\to B}^t} v(f)
\]

o

\[
\text{TCP\_FLAGS\_FREQ}(e) = \frac{1}{n_e} \sum_{f \in \mathcal{F}_{A\to B}^t} v(f)
\]

### 4.5. Protocolos y puertos

- `PROTOCOL` (IP), `L7_PROTO` (aplicación):
  - codificar como **one-hot** o **embedding** por categoría,
  - luego agregar por **media** (representa mezcla media de protocolos) o por suma (conteo).

- `L4_SRC_PORT`, `L4_DST_PORT`:
  - preferible mapear cada puerto a **categoría** (e.g., `HTTP`, `HTTPS`, `DNS`, `other_high_port`, etc.),
  - luego one-hot + sum / mean como en otros categóricos.

### 4.6. Features indicadoras para ICMP/DNS

Campos como:

- `ICMP_TYPE`, `ICMP_IPV4_TYPE`
- `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`

se usan sólo si aplica a ese flow. Para diferenciarlos de “0 = valor real”:

- añadir flags binarios:
  - `has_icmp = 1` si el flow es ICMP,
  - `has_dns = 1` si el flow es DNS,
- y dejar los campos a 0 cuando no aplica.

---

## 5. Features de nodos

Cada nodo \(v \in V_t\) representa una IP en la ventana \(W_t\).

### 5.1. Nodo mínimo (baseline)

Versión mínima (para empezar):

- Todos los nodos tienen el mismo vector inicial (ej.: `[1]`).
- La GNN aprende a partir de:
  - la estructura del grafo,
  - y los features de las aristas.

### 5.2. Nodo enriquecido (recomendado)

Para mejor rendimiento y explicabilidad, definir features agregadas por IP:

Sea \(v\) una IP en \(V_t\).

- **Grados**:
  - `deg_in(v)` = número de edges entrantes,
  - `deg_out(v)` = número de edges salientes.

- **Volumen** (suma sobre todos los edges incidentes en la ventana):
  - `bytes_in_total(v)` = suma de `IN_BYTES`/`OUT_BYTES` de flows que tienen `DST = v`,
  - `bytes_out_total(v)` = suma de `IN_BYTES`/`OUT_BYTES` de flows que tienen `SRC = v`,
  - `ratio_bytes_out_in = bytes_out_total / (bytes_in_total + ε)`.

- **Diversidad**:
  - `num_peers(v)` = número de IPs distintas con las que se comunica,
  - `num_ports_src(v)` = cantidad de puertos origen distintos usados por v,
  - `num_ports_dst(v)` = cantidad de puertos destino distintos al hablar con v.

- **Distribución de protocolos**:
  - proporción de flows TCP/UDP/ICMP para v como src/dst (one-hot promediada).

### 5.3. Embedding de identidad de IP (opcional pero potente)

Mantener una tabla de embeddings para todas las IPs globales:

- `ip_embedding[global_node_id] ∈ R^d`

En cada ventana, el nodo correspondiente recibe:

- concatenación de:
  - embedding de identidad propiamente dicho,
  - + features agregadas de la ventana.

Esto ayuda a capturar roles persistentes a lo largo del tiempo (ej. servidor importante vs host marginal).

---

## 6. IDs de nodos y temporalidad

### 6.1 Mapeo global

Mantener una tabla global:

- `ip_to_global_id: IPv4 → entero`

Uso:

- Cada vez que aparece una IP nueva en cualquier ventana, se le asigna un nuevo `global_id`.

### 6.2 Grafos por ventana

- Cada \(G_t\) usa sólo los nodos **activos**.
- Los índices dentro del grafo son **locales**, pero podemos almacenar el `global_id` como metadato.

### 6.3 GNN + LSTM

- La GNN produce embedding del grafo \(h_t\).
- El LSTM procesa la secuencia \(\{h_t\}\).

> El LSTM aprende la **dinámica de la red en el tiempo**;  
> no requiere que “el nodo 0” signifique lo mismo entre ventanas.  

Opcionalmente (más avanzado):

- Se puede mantener un estado por nodo a través de ventanas, usando su `global_id`.  
  Pero esto no es obligatorio para una primera versión; basta con la secuencia de grafos.

---

## 7. Etiquetado

### 7.1 Etiqueta por ventana (para el modelo principal)

Sea \(\mathcal{F}^t\) el conjunto de flows en \(W_t\):

\[
y_t =
\begin{cases}
1 & \text{si } \exists f\in\mathcal{F}^t\ \text{malicioso}\\
0 & \text{si no}
\end{cases}
\]

Esta etiqueta se usa para:

- entrenamiento de un clasificador binario a nivel grafo (ataque / no ataque),
- con foco en medir **early detection** respecto a ventanas que contienen flows de tipo *Impact*.

### 7.2 Etiquetas por flow (conservadas)

Cada flow mantiene su etiqueta original:

- `flow_label[flow_id] ∈ {benign, attack_type_1,...}`

### 7.3 Etiquetas por edge (agregadas pero SIN perder las originales)

Para cada edge \(e\) en ventana \(t\):

- guardar lista de flows subyacentes:

edge_flows[(t, edge_id)] = [flow_id1, flow_id2, ...]


- mantener también sus etiquetas:

edge_flow_labels[(t, edge_id)] = [flow_label(flow_id1), ...]


> No forzamos una única etiqueta por edge.  
> El post-procesamiento decidirá:
> - tipo de ataque predominante,
> - mapeo a MITRE,
> - o análisis de mezcla.

---

## 8. Trazabilidad y atención (Multi-head GAT)

Durante la construcción de grafos:

- Para cada arista \(e = (A,B)\) en ventana \(W_t\), guardar:

  - la lista de IDs de flows originales que se agregaron:
    - `edge_flows[(t, edge_id)] = [flow_id_1, flow_id_2, ...]`
  - y sus etiquetas `flow_label` correspondiente.

Esto no entra como feature en la GNN, pero se almacena:

- en estructuras auxiliares,
- para usarse luego en análisis / modelos secundarios.


Si la última capa de la GNN es una **GAT** sobre nodos o edges:

- La GAT produce:
  - pesos de atención \(\alpha_{ij}\) sobre edges \(i \to j\),
  - o sobre nodos, según el diseño.

Con la trazabilidad `edge_flows[(t, edge_id)]`:

- se puede:
  - identificar qué aristas tienen mayor atención en una ventana sospechosa,
  - extraer los flows correspondientes,
  - ver sus etiquetas originales,
  - y así interpretar:
    - qué IPs,
    - qué tipo de tráfico,
    - qué tipo de ataque,
    - mapear luego a mitre
  están siendo considerados más relevantes por el modelo.

> **Importante**: para esto no hace falta modificar la construcción de grafos más allá de mantener la tabla `edge_flows` y `flow_labels`.  
> Pero es fundamental **no perder esa relación** durante el preprocesamiento.

---

## 9. Consideraciones Adicionales

### 9.1. Normalización de features

Antes de entrenar la GNN:

- aplicar:
  - `log1p` en features de gran rango (bytes, pkts, retransmisiones),
  - normalización (z-score) a todas las features numéricas (edge y node),
  - opcionalmente, normalización por ventana
  - evitar normalización con información del futuro (respetar orden temporal)

### 9.2. Grafo vacío o muy pequeño

Si una ventana \(W_t\) tiene:

- muy pocos flows (o ninguno), el grafo puede no ser informativo.

Estrategia:

- descartar ventanas sin flows,
- o marcar explicitamente con `y_t = 0` y un grafo trivial (según la métrica que quieras usar).

### 9.3. Split de train/val/test

Para preservar la temporalidad:

- dividir por tiempo (no aleatoriamente por flows):
  - primeras ventanas → train
  - siguientes → validación
  - últimas → test

### 9.4. Desbalance de clases

Es habitual que haya muchas más ventanas benignas que maliciosas:

- usar:
  - ponderación de pérdida (class weights),
  - y/o técnicas de submuestreo/sobremuestreo a nivel ventana.

---

## 10. Resumen Esquemático

1. **Construir ventanas deslizantes**:
   - construir \(W_t\) con `WINDOW_SIZE`, `WINDOW_STEP`,
   - asignar flows a ventanas por `FLOW_START_MILLISECONDS`.

2. **Crear grafo dirigido por ventana \(G_t\)**:
   - nodos: IPs activas en la ventana,
   - edges dirigidos/bidireccionales: `src→dst` si hay al menos un flow en esa dirección.

3. **Edge features**:
   - sumar cantidades (bytes, pkts, retransmisiones),
   - media/std/min/max para propiedades de flows (duraciones, IATs, TTL, tamaños),
   - one-hot + sum/mean para categóricas (protocolos, puertos, flags, ICMP/DNS),
   - incluir `flow_count`.

4. **Node features**:
   - baseline: vector constante,
   - enriquecido: grados, volúmenes agregados, diversidad de peers/puertos, distribución de protocolos,
   - opcional: embedding de identidad por IP (`global_id`).

5. **IDs de nodos globales**:
   - mapear cada IP a un `global_id`,
   - usar `global_id` como referencia estable a lo largo de ventanas.

6. **Etiquetado de ventanas**:
   - binario: `y_t = 1` si algún flow malicioso en la ventana, si no `0`.

7. **Trazabilidad para modelos secundarios / atención**:
   - guardar para cada (ventana, edge) la lista de `flow_id` asociados,
   - conservar `flow_label` por flow,
   - usar esto después para:
     - análisis de atención,
     - modelos secundarios (ej. clasificación por flow o por tipo de ataque),
     - mapeo a MITRE.

8. **Mantener**:
   - `flow_id`,
   - `flow_label`,
   - `edge_flows`,
   - `edge_flow_labels`,  
   
9. **Entrenar** GNN→LSTM para \(y_t\),  

---

## 11. Manejo de valores centinela y campos “no aplicables”

El dataset NF-v3 no contiene valores `NaN` explícitos; sin embargo, emplea **valores centinela**
(especialmente `0`) para representar campos *no aplicables* —por ejemplo:

- métricas DNS en flows que no son DNS,
- métricas ICMP en flows que no son ICMP,
- métricas TCP en flows que no son TCP, etc.

Estos valores **no deben interpretarse como mediciones reales**, sino como ausencia
semántica de información. Para evitar sesgos durante la agregación, se sigue el
procedimiento siguiente.

### 11.1 Identificación de “no aplica”

Se definen flags binarios por flow que indican cuándo un campo tiene sentido:

- `is_dns      = (DNS_QUERY_TYPE != 0)`
- `has_dns_ans = (DNS_TTL_ANSWER != 0)`
- `is_icmp     = (ICMP_TYPE != 0)`  
  (recordar que `ICMP_TYPE = type * 256 + code`)
- `is_tcp      = (PROTOCOL == 6)`  
  (TCP)
- otros flags similares según corresponda.

Estos flags se utilizan tanto para filtrado como para agregación.

### 11.2 Remapeo a NaN para agregación “consciente de NaN”

Para cualquier característica que **no aplica** en un flow:

- su valor se remapea temporalmente a `NaN`,
- y luego se utilizan funciones de agregación `nan-aware`:

- `nansum`, `nanmean`, `nanstd`, `nanmin`, `nanmax`.

Ejemplo conceptual (DNS TTL):

- si `DNS_TTL_ANSWER == 0` → remapear a `NaN`,
- al agregar en un edge:

> `ttl_avg_edge = nanmean(DNS_TTL_ANSWER_en_flows_del_edge)`

Esto evita que los ceros artificiales arrastren las medias o desvíos.

### 11.3 Máscaras para diferenciar “valor real” vs “ausente”

Cuando todos los flows de un edge tienen “no aplica” para una característica:

- la agregación devuelve `NaN`.

En ese caso:

- se imputa un valor neutro (típicamente `0`),
- **pero se añade un flag** que indica ausencia:

- `edge_has_dns = 1` si al menos un flow del edge es DNS, en otro caso `0`,
- `edge_is_icmp`, `edge_is_tcp`, etc.

Estos flags se concatenan al vector final de features del edge,
de modo que el modelo puede aprender la diferencia entre:

- “0 porque no aplica”  
y  
- “0 porque el valor real es 0”.

### 11.4 Impacto en nodos

Cuando se generan features agregadas a nivel nodo (dentro de la ventana):

- se aplican las mismas reglas:
  - remapeo a `NaN` para “no aplica”,
  - uso de `nan-*`,
  - y flags binarios de aplicabilidad cuando sea relevante.

### 11.5 Documentación en resultados

En los análisis exploratorios verificamos que:

- el dataset no contiene `NaN` explícitos,
- pero posee valores centinela frecuentes (`0`, códigos combinados),
- por lo que el pipeline adopta un tratamiento específico para preservar
la semántica “no aplica” sin introducir sesgo en las agregaciones.

Este mecanismo garantiza estabilidad numérica y evita que
los valores centinela dominen las estadísticas agregadas en edges y nodos.

---

## 12. Manejo de Campos Específicos de Protocolo

El dataset CIC-IDS2018-v3 presenta particularidades en el uso de campos
específicos de protocolo que requieren tratamiento cuidadoso:

#### DNS (DNS_QUERY_TYPE, DNS_TTL_ANSWER)
Análisis del dataset reveló que 0.28% de flows tienen `DNS_QUERY_TYPE != 0`
pero no usan puerto 53. Estos casos corresponden a:
- DNS sobre HTTPS (DoH) en puerto 443
- DNS sobre TLS (DoT) en puerto 853  
- Potencial DNS tunneling (ataque)

**Estrategia adoptada:** Considerar DNS cualquier flow con `DNS_QUERY_TYPE != 0`,
independiente del puerto. Esto captura ataques de tunneling mientras mantiene
cobertura de tráfico DNS legítimo en puertos no estándar.

#### ICMP (ICMP_TYPE)
Análisis reveló 1.5M flows (7%) con `ICMP_TYPE != 0` pero `PROTOCOL != 1`.
Investigación detallada mostró:
- 99% de estos flows son TCP (PROTOCOL=6)
- Los valores de ICMP_TYPE correlacionan con puertos TCP
- Los valores están fuera del rango válido ICMP (0-18)

**Conclusión:** ICMP_TYPE en flows no-ICMP es un artefacto de la herramienta
de captura (nfDPI), posiblemente por reutilización interna del campo.

**Estrategia adoptada:** Considerar ICMP SOLO flows con:
- `PROTOCOL == 1` (ICMPv4) o  
- `PROTOCOL == 58` (ICMPv6)

Ignorar valores de ICMP_TYPE en otros protocolos.

#### Agregación con Filtros
Para cada edge (par de IPs), se:

1. **Identifica flows relevantes** según protocolo real (no campos derivados)
2. **Calcula estadísticas** solo sobre flows donde el campo tiene semántica válida
3. **Añade flags binarios** indicando presencia de cada tipo de tráfico
4. **Incluye ratios** de composición de protocolo por edge

Ejemplo (DNS):
```python
# Filtrar flows DNS válidos
dns_flows = flows[(flows['DNS_QUERY_TYPE'] != 0)]

# Agregar (si hay flows DNS)
if len(dns_flows) > 0:
    dns_ttl_mean = dns_flows['DNS_TTL_ANSWER'].mean()
    edge_has_dns = 1.0
else:
    dns_ttl_mean = 0.0
    edge_has_dns = 0.0
```

Esta aproximación evita contaminación por valores centinela mientras
preserva poder discriminativo para detección de ataques.

#### Validación
Verificación cruzada entre campos confirmó coherencia en 99.7% de flows,
con excepciones explicables por protocolos avanzados (tunneling, encapsulación).

----

Este `.md` debería servir como especificación completa para implementar el pipeline de construcción de grafos y justificarlo en la memoria de doctorado (capítulo de metodología y apéndice técnico).





