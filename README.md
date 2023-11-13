# Estación Aprendizaje Ranking API

Este repositorio alberga un algoritmo diseñado para predecir la dificultad adecuada del próximo conjunto de problemas que un estudiante deberá resolver en la aplicación [Estación Aprendizaje](https://github.com/erickborquez/estacion-aprendizaje). La predicción se basa en el historial de resultados de sets de problemas previos, permitiendo una adaptación personalizada a las habilidades y progreso del usuario.

### Filosofía

- **Inicialización**: Inicie a nuevos usuarios en un nivel de dificultad moderado (por ejemplo, 2) como punto de referencia para ajustar.

- **Evaluación del Rendimiento**:Evalúe a los usuarios según sus puntajes más recientes, centrándose en las tendencias en lugar de los puntajes individuales.

- **Reconocimiento de Zona de Confort**: Si un usuario obtiene puntajes consistentemente altos pero no muestra mejora, mantenga el nivel de dificultad actual hasta que demuestren estar listos para progresar.

- **Elevación Progresiva**: Si un usuario obtiene puntajes consistentemente altos y muestra mejora, incremente gradualmente la dificultad.

- **Adaptación a la Regresión**: Si los puntajes de un usuario comienzan a disminuir significativamente, reduzca el nivel de dificultad para facilitar el aprendizaje.

- **Tasa de Cambio**: La tasa de ajuste de dificultad debe ser proporcional a la tasa de mejora o regresión del usuario.

- **Requisito de Consistencia**: Un cambio en el nivel de dificultad solo debe activarse por patrones de rendimiento consistentes, no por puntajes aislados.

- **Meseta de Rendimiento**: Si el rendimiento de un usuario se estanca, mantenga el nivel de dificultad actual para dominar el contenido antes de avanzar.

Estos principios proporcionan un marco definido para el algoritmo, equilibrando el desafío y la accesibilidad en función del rendimiento individual.

### Construcción del Algoritmo
El Algoritmo de Adaptación se construye utilizando XGBoost junto con Grid Search para ajuste de hiperparámetros. La elección de este enfoque se basa en la capacidad de XGBoost para manejar el sobreajuste, procesar eficientemente conjuntos de datos extensos, gestionar datos faltantes y su flexibilidad y capacidad de optimización. La combinación con Grid Search asegura una afinación efectiva de los hiperparámetros para mejorar el rendimiento del modelo.


## API Pública

/home
- Método: GET
- Descripción: Devuelve un mensaje de bienvenida.
- Parámetros: Ninguno

/predict
- Método: POST
- Descripción: Devuelve la predicción de dificultad para el próximo set de problemas.
- Parámetros: Lista de resultados de problemas previos (0 a 100), separados por comas:
```json
{ "scores": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1] }
```


## Configuración del Entorno de Desarrollo

1. IDE Recomendado: PyCharm
Se recomienda utilizar PyCharm como entorno de desarrollo integrado (IDE) para facilitar el desarrollo y mantenimiento del proyecto. Asegúrate de tenerlo instalado y configurado en tu sistema.

2. Activar el Entorno Virtual
En el terminal, activa el entorno virtual según tu sistema operativo:
   - En Mac/Linux, utiliza: `source venv/bin/activate`
   - En Windows, utiliza: `venv\Scripts\activate`
3. Instalación de Dependencias
Ejecuta el siguiente comando para instalar las dependencias necesarias. Asumiendo que estás utilizando pip como gestor de paquetes:

```bash
pip install -r requirements.txt
```

## Ejecución del Proyecto
**Iniciar la API de Flask**: Ejecuta el archivo app.py para iniciar la API de Flask. Asegúrate de que el entorno virtual esté activo antes de ejecutar el siguiente comando:
```bash
python app.py
```

**Actualización del Modelo**: Si se realizan actualizaciones en el archivo model.py, asegúrate de ejecutarlo para que el modelo se actualice. Usa el siguiente comando:

bash
Copy code
python model.py

## Hosting
Para conectar la aplicación Flask a Internet, se recomienda el uso de ngrok. La aplicación está configurada para ejecutarse en el puerto 3001. Utiliza el siguiente comando de ngrok para exponer tu aplicación:
```bash
ngrok http --domain=<domain> 3001
```
Reemplaza <domain> con el dominio que desees asignar a tu aplicación.

¡Con estos pasos, tu entorno de desarrollo debería estar listo y la aplicación Flask accesible en línea a través de ngrok!





