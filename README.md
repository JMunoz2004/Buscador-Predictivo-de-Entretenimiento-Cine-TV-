# ¿Qué Veo Hoy? — Buscador Predictivo de Entretenimiento

## Descripción
"¿Qué Veo Hoy?" es una aplicación web interactiva diseñada para eliminar la parálisis por análisis y la fatiga de elección al buscar contenido en plataformas de streaming (Netflix, HBO Max, Disney+, Prime Video). A partir del tiempo disponible del usuario, su estado de ánimo, época preferida y suscripciones activas, la aplicación predice y recomienda mediante aprendizaje automático la película o serie con mayor porcentaje de afinidad (*match score*) para ese momento concreto.

## Objetivos
1. Consolidar y procesar un dataset unificado de metadatos cinematográficos y disponibilidad en plataformas de streaming.
2. Entrenar y evaluar un modelo de recomendación híbrido (filtrado basado en contenido y regresión) que prediga la satisfacción del usuario.
3. Desplegar una interfaz web pública, intuitiva y accesible desde cualquier dispositivo.
4. Implementar un flujo de desarrollo continuo mediante actualizaciones periódicas en el repositorio a lo largo del semestre.

## Plan de trabajo inicial

| Fase | Tareas principales | Plazo estimado |
| :--- | :--- | :--- |
| **Fase 1: Datos y EDA** | Extracción de conjuntos de datos (MovieLens/TMDb), limpieza, imputación de nulos y análisis exploratorio de datos. | Semanas 1 - 2 |
| **Fase 2: Feature Engineering y Modelo** | Vectorización de características, diseño del sistema de recomendación híbrido y ajuste del modelo. | Semanas 3 - 4 |
| **Fase 3: Desarrollo Backend / API** | Construcción de la API para prestar servicio a las predicciones del modelo. | Semanas 5 - 6 |
| **Fase 4: Frontend y Despliegue** | Diseño de la interfaz interactiva y despliegue público en producción. | Semanas 7 - 8 |
