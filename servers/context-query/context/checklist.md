<!-- SECTION_ID: business_model -->
# Modelo de Negocio - Yari-system

## Visión General
Yari-System es un sistema integral de gestión médica diseñado inicialmente para consultorios privados, con capacidad de escalar a clínicas completas y hospitales. El modelo de negocio se basa en software como servicio (SaaS) con implementación local.

## Fuentes de Ingreso
- **Licencias de software**: Venta de licencias perpetuas con mantenimiento anual
- **Implementación y capacitación**: Servicios profesionales para setup inicial
- **Soporte técnico**: Contratos de mantenimiento y actualizaciones
- **Personalización**: Desarrollo de módulos específicos según requerimientos

## Valor Diferencial
- **Arquitectura modular**: Permite expansión gradual sin refactorización completa
- **Stack maduro**: Django + PostgreSQL para escalabilidad empresarial
- **Cumplimiento médico**: Manejo adecuado de datos sensibles de salud
- **Localización**: Soporte completo para español e idioma médico

## Mercado Objetivo
- **Consultorios médicos privados**: Especialistas individuales o grupos pequeños
- **Clínicas**: Centros médicos con múltiples especialidades
- **Hospitales pequeños**: Instalaciones con servicios básicos de internación
- **Grupos médicos**: Redes de atención médica coordinada

## Escalabilidad
El sistema está diseñado para crecer desde 1 usuario hasta miles, manteniendo la misma arquitectura base pero añadiendo módulos especializados según las necesidades.
<!-- SECTION_ID: business_model -->

<!-- SECTION_ID: product_vision -->
# Visión de Producto - Yari-System

## Objetivos Trimestrales 2025
- **Q1**: Completar expansión a clínica completa con hospitalización
- **Q2**: Implementar integración con equipos médicos (PACS, monitores)
- **Q3**: Lanzar app móvil para personal médico
- **Q4**: Implementar inteligencia artificial para diagnósticos asistidos

## Métricas Clave
- **Satisfacción del usuario**: >95% (medido por NPS)
- **Tiempo de respuesta**: <2 segundos para operaciones críticas
- **Disponibilidad**: 99.9% uptime
- **Adopción de módulos**: >80% de módulos implementados en uso activo

## Hoja de Ruta Tecnológica
### Fase Actual: Consolidación (2025)
- Migración completa a PostgreSQL
- Implementación de APIs REST
- completar módulos críticos
- Contenedores Docker para despliegue

### Fase Futura: Innovación (2026)
- Integración con IA para análisis de imágenes médicas
- Telemedicina integrada
- Blockchain para historial médico inmutable
- IoT para monitoreo de pacientes
<!-- SECTION_ID: product_vision -->

<!-- SECTION_ID: tech_architecture -->
# Arquitectura Técnica - Yari-System

## Stack Tecnológico Principal
- **Backend**: Django 5.0.6 (Python 3.11+)
- **Base de Datos**: PostgreSQL (producción) / SQLite (desarrollo)
- **Frontend**: Django Templates + Tailwind CSS + Crispy Forms
- **APIs**: Django REST Framework (implementación progresiva)
- **Servidor**: Gunicorn + Nginx (producción)

## Patrones de Arquitectura Implementados
### MVC (Model-View-Controller)
- **Models**: Relaciones complejas con ForeignKey y OneToOneField
- **Views**: Function-based views con decoradores de autenticación
- **Templates**: Sistema de herencia con componentes reutilizables

### Repository Pattern
- Queries optimizadas con Q objects
- Unión de querysets para búsquedas multi-campo
- Managers personalizados para lógica de negocio

### Observer Pattern
- Signals para actualización automática de estadísticas
- Historial de modificaciones transparente
- Integración entre módulos

### Strategy Pattern
- Múltiples estrategias de facturación por tipo de paciente
- Métodos de pago configurables
- Flujos de trabajo adaptables

## Estructura de Base de Datos
### Entidades Core
```
User (Django Auth)
├── Paciente (1:N con Citas, Facturas, HistorialMedico)
├── Medico (1:N con Citas)
├── Cita (relacionada con Paciente, Medico, HistoriaClinica)
├── HistoriaClinica (OneToOne con Paciente)
├── Factura (relacionada con Paciente, MovimientoFinanciero)
└── MovimientoFinanciero (ingresos/egresos)
```

### Relaciones Clave
- Paciente → Citas (1:N)
- Paciente → Facturas (1:N)
- Paciente → HistorialMedico (1:1)
- Cita → HistoriaClinica (opcional)
- Factura → MovimientoFinanciero (1:1)

## Límites del Sistema
### Rendimiento
- **Consultas simultáneas**: Máximo 100 usuarios concurrentes por instancia
- **Tamaño de base de datos**: Optimizado para hasta 1M de registros
- **Tiempo de respuesta**: <500ms para operaciones críticas

### Escalabilidad
- **Horizontal**: Posibilidad de múltiples instancias con balanceo de carga
- **Vertical**: Optimización para servidores con 8+ CPU cores y 32GB+ RAM
- **Datos**: Particionamiento por fecha para tablas grandes

### Seguridad
- **Autenticación**: Sistema de roles granular con permisos específicos
- **Datos sensibles**: Encriptación de información médica confidencial
- **Auditoría**: Log completo de todas las operaciones críticas
<!-- SECTION_ID: tech_architecture -->

<!-- SECTION_ID: coding_conventions -->
# Convenciones de Código - Yari-System

## Estructura de Archivos
### Apps Django
```
mi_app/
├── __init__.py
├── admin.py          # Configuración de Django Admin
├── apps.py           # Configuración de la app
├── models.py         # Modelos de datos
├── views.py          # Lógica de vistas (function-based)
├── urls.py           # Definición de URLs
├── forms.py          # Formularios Django
├── tests.py          # Tests unitarios
├── managers.py       # Managers personalizados (opcional)
└── utils.py          # Utilidades específicas (opcional)
```

## Naming Conventions
### Python
- **Clases**: PascalCase (MiClase, PacienteForm)
- **Funciones/Métodos**: snake_case (crear_paciente, get_absolute_url)
- **Variables**: snake_case (nombre_completo, fecha_nacimiento)
- **Constantes**: UPPER_SNAKE_CASE (MAX_RETRIES, DEFAULT_TIMEOUT)

### Base de Datos
- **Tablas**: app_modelo (pacientes_paciente, citas_cita)
- **Campos**: snake_case (nombre, fecha_nacimiento, tipo_documento)
- **Foreign Keys**: modelo_id (paciente_id, medico_id)

### URLs
- **URLs principales**: kebab-case (lista-pacientes, crear-cita)
- **Parámetros**: <int:pk> para IDs, <str:slug> para slugs

## Estilos de Código
### Imports
```python
# Estándar library imports primero
import os
import json
from datetime import datetime

# Django imports
from django.db import models
from django.contrib.auth.models import User
from django.shortcuts import render, redirect

# Third-party imports
import requests

# Local imports
from .models import Paciente
from ..utils import format_currency
```

### Modelos
```python
class Paciente(models.Model):
    TIPO_DOCUMENTO_CHOICES = [
        ("CC", "Cédula de ciudadanía"),
        ("TI", "Tarjeta de identidad"),
        ("CE", "Cédula de extranjería"),
        ("PA", "Pasaporte"),
    ]

    # Campos básicos primero
    nombre = models.CharField(max_length=100)
    apellido = models.CharField(max_length=100)

    # Campos específicos después
    tipo_documento = models.CharField(
        max_length=2,
        choices=TIPO_DOCUMENTO_CHOICES,
        default="CC"
    )
    numero_documento = models.CharField(max_length=50, unique=True)

    # Relaciones al final
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name='pacientes_creados'
    )

    class Meta:
        verbose_name = "Paciente"
        verbose_name_plural = "Pacientes"
        ordering = ['apellido', 'nombre']

    def __str__(self):
        return f"{self.nombre} {self.apellido}"

    def nombre_completo(self):
        return f"{self.nombre} {self.apellido}"
    nombre_completo.short_description = "Nombre Completo"
```

### Vistas
```python
@login_required
def lista_pacientes(request):
    """Lista todos los pacientes con filtros opcionales"""
    busqueda = request.GET.get('busqueda', '')

    if busqueda:
        pacientes = Paciente.objects.filter(
            Q(nombre__icontains=busqueda) |
            Q(apellido__icontains=busqueda) |
            Q(numero_documento__icontains=busqueda)
        )
    else:
        pacientes = Paciente.objects.all()

    pacientes = pacientes.order_by('apellido', 'nombre')

    return render(request, 'pacientes/lista_pacientes.html', {
        'pacientes': pacientes,
        'busqueda': busqueda
    })

@login_required
def crear_paciente(request):
    """Crear nuevo paciente"""
    if request.method == 'POST':
        form = PacienteForm(request.POST)
        if form.is_valid():
            paciente = form.save(commit=False)
            paciente.created_by = request.user
            paciente.save()

            messages.success(request, 'Paciente creado correctamente')
            return redirect('pacientes:lista_pacientes')
    else:
        form = PacienteForm()

    return render(request, 'pacientes/formulario_paciente.html', {
        'form': form,
        'titulo': 'Crear Paciente'
    })
```

## Testing
### Estructura de Tests
```python
# tests.py
from django.test import TestCase
from django.contrib.auth.models import User
from .models import Paciente

class PacienteModelTest(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('testuser', 'test@example.com', 'password')
        self.paciente = Paciente.objects.create(
            nombre='Juan',
            apellido='Pérez',
            numero_documento='12345678'
        )

    def test_paciente_creation(self):
        self.assertEqual(self.paciente.nombre, 'Juan')
        self.assertEqual(self.paciente.apellido, 'Pérez')

    def test_paciente_str(self):
        self.assertEqual(str(self.paciente), 'Juan Pérez')
```

### Ejecución de Tests
```bash
# Todos los tests
python manage.py test

# Tests de una app específica
python manage.py test pacientes

# Tests de una clase específica
python manage.py test pacientes.PacienteModelTest

# Con coverage
coverage run manage.py test
coverage report
```
<!-- SECTION_ID: coding_conventions -->

<!-- SECTION_ID: workflow -->
# Flujo de Trabajo - Yari-System

## Desarrollo de Características
### 1. Planificación
- **Issue Creation**: Crear issue detallado en GitHub con aceptación criteria
- **Branch Creation**: `feature/nombre-descriptivo` desde `main`
- **Design Review**: Discutir implementación con el equipo

### 2. Desarrollo
- **TDD Approach**: Escribir tests antes del código cuando sea posible
- **Incremental Commits**: Commits pequeños con mensajes descriptivos
- **Code Reviews**: Pull requests requieren aprobación de al menos 1 reviewer

### 3. Testing
- **Unit Tests**: Cobertura >80% para lógica crítica
- **Integration Tests**: Verificar interacción entre módulos
- **Manual Testing**: QA por usuario final para UX

### 4. Despliegue
- **Staging**: Deploy automático a staging desde `main`
- **Production**: Deploy manual con checklist de verificación
- **Rollback Plan**: Estrategia clara para reversiones

## Control de Versiones
### Ramas
- **main**: Código de producción estable
- **develop**: Integración de features
- **feature/***: Nuevas funcionalidades
- **bugfix/***: Corrección de bugs
- **hotfix/***: Fixes críticos en producción

### Commits
```bash
# Formato: tipo(alcance): descripción

feat(pacientes): agregar validación de documento único
fix(facturacion): corregir cálculo de IVA
docs(readme): actualizar guía de instalación
style(forms): formatear código con black
refactor(models): optimizar queries de pacientes
test(citas): agregar tests para validaciones
```

## CI/CD Pipeline
### GitHub Actions
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: python manage.py test --verbosity=2
    - name: Run linting
      run: flake8
```

### Despliegue
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  release:
    types: [published]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to server
      run: |
        echo "Desplegando a producción..."
        # Comandos de despliegue específicos
```

## Code Review Process
### Checklist de PR
- [ ] **Funcionalidad**: ¿Hace lo que dice el título?
- [ ] **Tests**: ¿Tiene tests apropiados?
- [ ] **Documentación**: ¿Actualiza docs si es necesario?
- [ ] **Estilo**: ¿Sigue las convenciones del proyecto?
- [ ] **Performance**: ¿No degrada el rendimiento?
- [ ] **Seguridad**: ¿No introduce vulnerabilidades?

### Aprobación
- **1 approval** mínimo para features pequeñas
- **2 approvals** para cambios críticos (models, auth, payments)
- **QA approval** requerido para cambios de UI/UX

## Ambiente de Desarrollo
### Setup Local
```bash
# Clonar repositorio
git clone https://github.com/tu-org/softmedic.git
cd softmedic

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar base de datos
cp .env.example .env
python manage.py migrate

# Crear superusuario
python manage.py createsuperuser

# Ejecutar servidor
python manage.py runserver
```

### Herramientas Recomendadas
- **Editor**: VS Code con extensiones Python
- **Linting**: flake8, black
- **Testing**: pytest con coverage
- **API Testing**: Postman o HTTPie
- **DB Viewer**: DBeaver o pgAdmin

## Monitoreo y Alertas
### Métricas a Monitorear
- **Performance**: Response times, throughput
- **Errors**: 500 errors, exceptions
- **Usage**: Active users, feature adoption
- **Business**: Conversion rates, revenue metrics

### Alertas
- Response time > 5 segundos
- Error rate > 5%
- Database connection issues
- Disk space < 10%
<!-- SECTION_ID: workflow -->

<!-- SECTION_ID: constraints -->
# Restricciones y Límites - Yari-System

## Tecnologías Prohibidas
### Frameworks y Librerías
- **No usar**: FastAPI, Flask (solo Django para consistencia)
- **No usar**: SQLAlchemy (solo Django ORM)
- **No usar**: Jinja2 (solo Django Templates)
- **Evitar**: Librerías con licencias copyleft restrictivas

### Patrones Anti-Patrón
- **No usar**: Raw SQL queries (excepto para optimizaciones críticas)
- **No usar**: Model managers complejos (preferir métodos en modelos)
- **Evitar**: Vistas genéricas over-engineered
- **No usar**: Mixins múltiples en una sola clase

## Restricciones de Arquitectura
### Base de Datos
- **Solo PostgreSQL** en producción (no MySQL, Oracle, etc.)
- **No triggers** en base de datos (lógica en aplicación)
- **No stored procedures** (mantenibilidad)
- **Foreign keys required** para integridad referencial

### APIs y Integraciones
- **Solo Django REST Framework** para APIs
- **No GraphQL** (REST es suficiente)
- **Autenticación JWT** solo donde sea necesario
- **Rate limiting** en todos los endpoints públicos

## Límites de Performance
### Base de Datos
- **Máximo 1000 registros** por query sin paginación
- **Índices obligatorios** en campos de búsqueda frecuentes
- **No joins excesivos** (máximo 3 tablas por query)
- **Cache obligatorio** para queries repetitivas

### Frontend
- **Máximo 5MB** por página (incluyendo assets)
- **No JavaScript pesado** (Django templates + HTMX preferido)
- **Responsive design** obligatorio
- **Accesibilidad WCAG 2.1** AA mínimo

### APIs
- **Timeout máximo 30 segundos** por request
- **Rate limit 100 requests/minuto** por usuario
- **Paginación obligatoria** para listas grandes
- **Versionado semántico** en URLs

## Restricciones de Seguridad
### Autenticación
- **Solo Django Auth** (no custom auth systems)
- **2FA obligatorio** para administradores
- **Sesiones expiran** en 24 horas máximo
- **Password policy** estricta (12+ caracteres, complejidad)

### Datos Sensibles
- **Encriptación AES-256** para datos médicos
- **Auditoría completa** de accesos a datos sensibles
- **Backup encriptado** obligatorio
- **No logs** con datos sensibles en claro

### Red y Conectividad
- **HTTPS obligatorio** en producción
- **HSTS header** configurado
- **CORS restringido** a dominios autorizados
- **CSP header** implementado

## Restricciones de Desarrollo
### Code Quality
- **Coverage mínimo 80%** para código crítico
- **Cyclomatic complexity < 10** por función
- **Máximo 300 líneas** por archivo
- **Máximo 50 líneas** por función

### Dependencias
- **Actualizaciones trimestrales** de dependencias
- **Security audits** mensuales
- **No dependencias abandonadas** (último commit < 6 meses)
- **Licencias compatibles** verificadas

### Deployment
- **Zero-downtime deployments** obligatorio
- **Blue-green strategy** para updates críticos
- **Database migrations** reversibles
- **Rollback automático** en caso de falla

## Límites de Escalabilidad
### Usuarios Concurrentes
- **Máximo 500 usuarios** por instancia Django
- **Balanceo de carga** requerido después de 200 usuarios
- **Cache distribuido** (Redis) obligatorio

### Datos
- **Máximo 10M registros** por tabla principal
- **Particionamiento** por fecha para datos históricos
- **Archiving** automático de datos antiguos
- **Backup incremental** diario

## Restricciones de Costo
### Infraestructura
- **Máximo $500/mes** por instancia en cloud
- **Auto-scaling** basado en métricas, no en costos
- **Reserved instances** para producción estable
- **Spot instances** solo para desarrollo/staging

### Desarrollo
- **Máximo 20% overhead** en tooling vs desarrollo
- **Herramientas open source** preferidas
- **Licencias razonables** solo cuando necesarias
- **ROI mínimo 3:1** para herramientas pagas

## Consecuencias de Violación
### Leves (advertencia)
- Código no sigue convenciones de estilo
- Tests con coverage baja (60-79%)
- Documentación incompleta

### Graves (rechazo de PR)
- Vulnerabilidades de seguridad introducidas
- Performance degradation >20%
- Breaking changes sin migración backward-compatible
- Dependencias con vulnerabilidades conocidas

### Críticas (reversión inmediata)
- Exposición de datos sensibles
- Sistema inoperable
- Pérdida de datos
- Violación de compliance médico/legal
<!-- SECTION_ID: constraints -->
