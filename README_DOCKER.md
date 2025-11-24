# Ejecutar Padel Analytics con Docker

Este proyecto ha sido dockerizado para facilitar su despliegue y ejecución en cualquier equipo con soporte para Docker y NVIDIA GPU.

## Prerrequisitos

1.  **Docker**: Tener instalado Docker Desktop o Docker Engine.
2.  **NVIDIA Drivers**: Tener los drivers de tu tarjeta gráfica instalados.
3.  **NVIDIA Container Toolkit**: Necesario para que Docker pueda acceder a la GPU.
    *   Guía de instalación: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Instrucciones

### 1. Preparar los pesos (Weights)
Asegúrate de que la carpeta `weights/` contiene todos los modelos necesarios (`.pt` files). Si no los tienes, descárgalos según las instrucciones del README principal.

### 2. Construir y Ejecutar

Usa `docker-compose` para construir la imagen y levantar el contenedor:

```bash
docker-compose up --build
```

### 3. Acceder a la Aplicación

Una vez que el contenedor esté corriendo, abre tu navegador y ve a:

http://localhost:8501

### Notas Importantes

*   **Persistencia**: Las carpetas `weights`, `examples` y `cache` están montadas como volúmenes. Esto significa que si añades videos a `examples` en tu máquina, aparecerán en el contenedor. Lo mismo aplica para los pesos y la caché.
*   **GPU**: El archivo `docker-compose.yml` está configurado para usar la GPU. Si no tienes GPU o no has configurado el NVIDIA Container Toolkit, el contenedor podría fallar o funcionar muy lento (si PyTorch hace fallback a CPU, aunque el código fuerza CUDA en algunos puntos).
