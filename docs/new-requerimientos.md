# Plan Maestro para el Módulo de Farmacia

## 1. Alcance y Objetivos
- Gestionar inventario farmacéutico incluyendo recepción por lotes, control de expiraciones y valorización por unidad/caja/mayor.
- Conectar farmacia con todos los departamentos para solicitudes, consumo y facturación.
- Automatizar alertas (stock bajo, vencimientos) y reportes de márgenes, costos y ganancias.

## 2. Arquitectura Propuesta
- **App `farmacia/`** con modelos claves:
  - `Medicamento`: ficha base con datos clínicos y comerciales.
  - `LoteMedicamento`: control de lotes, cantidades, costos, fechas.
  - `MovimientoInventario`: entradas/salidas/ajustes asociados a lotes.
  - `SolicitudMedicamento`: workflow de pedidos interdepartamentales.
  - `PrecioEspecial` y `ConfiguracionAlertas` para márgenes y umbrales.
- Integraciones con `pacientes`, `medicos`, `facturacion` y `correos` mediante FK y señales.

## 3. Flujos Clave
1. **Recepción de lotes**: registro por farmacia, generación de movimiento tipo entrada y validación de costos.
2. **Inventario dinámico**: dashboard con KPIs, lotes próximos a vencer, stock mínimo.
3. **Solicitudes de departamentos**: creación, aprobación por farmacia, seguimiento de estado y notificaciones.
4. **Venta a pacientes**: selección de lotes (FIFO), cálculo de precios por unidad/caja, margen según segmento, emisión de factura y envío de correo.
5. **Alertas programadas**: vencimientos, stock bajo, consumo inusual; distribución vía correo y badges en dashboard.

## 4. Interfaces y Permisos
- Vistas farmacia (dashboard, catálogo, lotes, movimientos, reportes) protegidas por rol.
- Secciones para solicitudes interdepartamentales con visibilidad según departamento.
- Panel administración para revisar precios, márgenes y facturación asociada.

## 5. Roadmap de Implementación
- **Fase 1 (Backend base)**: modelos, servicios de inventario, tests unitarios.
- **Fase 2 (UI Farmacia)**: dashboard, recepción de lotes, alertas básicas.
- **Fase 3 (Interdepartamental)**: solicitudes, notificaciones, workflow.
- **Fase 4 (Facturación y ventas)**: integración con `facturacion`, plantillas de correo.
- **Fase 5 (Alertas avanzadas / reporting)**: jobs programados, gráficos y KPIs.
- **Fase 6 (Finanzas y administración)**: exportaciones, conciliaciones, panel de ganancias.

## 6. Consideraciones
- Seguridad: permisos por rol, auditoría de movimientos, cifrado de datos sensibles.
- Calidad: pruebas unitarias/E2E, documentación `docs/modules/farmacia.md`.
- Observabilidad: logging de eventos críticos y monitoreo de tareas programadas.

## 7. Próximos Pasos
- Validar requerimientos con farmacia y administración.
- Definir roles de usuario para acceso al módulo.
- Priorizar MVP (Fases 1-3) antes de automatizaciones avanzadas.

# 🗺️ ROADMAP DETALLADO - INTEGRACIÓN VOCODE PARA INFORMES MÉDICOS

https://github.com/vocodedev/vocode-core

## 📋 **ANÁLISIS DEL CONTEXTO ACTUAL**

### **Módulos Existentes Relevantes:**
- **Historia Clínica** (`historia_clinica/views.py`, `historia_clinica/forms.py`)
- **Consultas Médicas** (ya existe el flujo)
- **Sistema de Autenticación y Roles** (para control de acceso)

### **Capacidades Actuales del Sistema:**
- Gestión completa de pacientes
- Registro de consultas médicas
- Sistema de citas y agenda
- Control de usuarios y permisos

---

## 🎯 **OBJETIVO ESTRATÉGICO**

**Implementar un sistema de transcripción y generación de informes médicos mediante conversación médico-IA que se integre perfectamente con los módulos existentes, manteniendo la seguridad y flujos actuales.**

---

## 🗂️ **ARQUITECTURA PROPUESTA**

### **1. NUEVO MÓDULO: `transcripcion_voz/`**

```
transcripcion_voz/
├── models.py          # Modelos para conversaciones e informes
├── views.py           # Vistas para gestión de transcripción
├── forms.py           # Formularios para edición de informes
├── services.py        # Lógica de integración con Vocode
├── utils.py           # Utilidades para procesamiento
└── urls.py            # Rutas del módulo
```

### **2. MODELOS DE DATOS NECESARIOS:**

```python
# Esquema conceptual (sin código)
TranscripcionConsulta:
- paciente (FK a Paciente)
- medico (FK a User)
- fecha_consulta
- audio_original (FileField)
- texto_transcrito (TextField)
- estado [pendiente, transcribiendo, revisión, aprobado]

InformeMedico:
- transcripcion (OneToOne a TranscripcionConsulta)
- contenido_estructurado (JSONField)
- borrador_ia (TextField)
- informe_final (TextField)
- editado_por (FK a User)
- fecha_aprobacion
```

---

## 🔄 **FLUJO DE TRABAJO COMPLETO**

### **FASE 1: CONFIGURACIÓN Y CAPTURA** (Semana 1-2)

```mermaid
flowchart TD
    A[Inicio Consulta Médica] --> B[Médico activa<br>grabación]
    B --> C[Sistema captura<br>audio en tiempo real]
    C --> D[Almacenamiento seguro<br>con cifrado]
    D --> E[Transcripción automática<br>con Vocode]
    E --> F[Generación borrador<br>por IA]
```

**Componentes:**
- **Servicio de Captura de Audio**: Integración con micrófono del consultorio
- **Almacenamiento Temporal**: Audio cifrado mientras se procesa
- **Conexión Vocode**: Configuración de APIs y seguridad

### **FASE 2: PROCESAMIENTO Y ESTRUCTURACIÓN** (Semana 3-4)

```mermaid
flowchart TD
    A[Texto Transcribido] --> B[Procesamiento con LLM<br>medicalizado]
    B --> C[Identificación de<br>secciones clínicas]
    C --> D[Extracción de<br>entidades médicas]
    D --> E[Generación de<br>borrador estructurado]
    E --> F[Presentación para<br>revisión médica]
```

**Procesamiento Inteligente:**
- **Motor de Entidades Médicas**: Síntomas, diagnósticos, tratamientos
- **Estructuración Automática**: Motivo consulta, antecedentes, exploración, etc.
- **Validación de Consistencia**: Verificación de datos críticos

### **FASE 3: REVISIÓN Y APROBACIÓN** (Semana 5-6)

```mermaid
flowchart TD
    A[Borrador de IA] --> B[Interfaz de<br>edición médica]
    B --> C[Médico revisa<br>y corrige]
    C --> D{¿Aprobar?}
    D -->|Sí| E[Guardar en<br>Historia Clínica]
    D -->|No| F[Edición<br>completa]
    F --> C
    E --> G[Notificación a<br>sistema principal]
```

**Interfaz de Revisión:**
- Editor enriquecido con sugerencias de IA
- Resaltado de términos médicos
- Validación de campos obligatorios
- Firma digital del médico

---

## 🔗 **INTEGRACIÓN CON MÓDULOS EXISTENTES**

### **1. CON HISTORIA CLÍNICA:**
- **Punto de Integración**: `historia_clinica/views.py` - método de creación de consultas
- **Flujo**: El informe aprobado se convierte en entrada de historia clínica automáticamente
- **Datos Compartidos**: Paciente, médico, fecha, diagnóstico, tratamiento

### **2. CON SISTEMA DE AUTENTICACIÓN:**
- **Control de Acceso**: Solo médicos pueden activar grabación
- **Auditoría**: Todo queda registrado con usuario y timestamp
- **Permisos**: Roles específicos para transcripción

### **3. CON MÓDULO DE PACIENTES:**
- **Vinculación**: Cada transcripción asociada a un paciente
- **Historial**: Acceso a informes desde ficha de paciente
- **Búsqueda**: Texto de transcripciones searchable

---

## 🛡️ **CONSIDERACIONES DE SEGURIDAD Y NORMATIVA**

### **Protección de Datos:**
- **Cifrado Extremo a Extremo**: Audio y texto en tránsito y reposo
- **Retención Limitada**: Audio original se elimina después de transcripción
- **Acceso por Roles**: Solo personal autorizado puede acceder
- **Auditoría Completa**: Log de quién accede y cuándo

### **Cumplimiento Normativo:**
- **HIPAA/GDPR**: Validación con proveedores de servicios (Vocode, STT, LLM)
- **Consentimiento**: Paciente debe autorizar grabación
- **Historial de Cambios**: Track de modificaciones en informes

---

## 📅 **PLAN DE IMPLEMENTACIÓN POR SPRINTS**

### **SPRINT 1: INFRAESTRUCTURA BASE** (2 semanas)
- [ ] Configuración entorno Vocode y APIs
- [ ] Modelos de datos para transcripciones
- [ ] Servicio básico de captura de audio
- [ ] Integración con autenticación existente

### **SPRINT 2: PROCESAMIENTO INTELIGENTE** (3 semanas)
- [ ] Integración STT (Speech-to-Text)
- [ ] Servicio de procesamiento con LLM
- [ ] Motor de estructuración de informes
- [ ] Almacenamiento seguro de datos

### **SPRINT 3: INTERFAZ Y FLUJOS** (2 semanas)
- [ ] Interfaz de revisión y edición
- [ ] Integración con historia clínica
- [ ] Flujo completo de aprobación
- [ ] Notificaciones y alertas

### **SPRINT 4: OPTIMIZACIÓN Y SEGURIDAD** (2 semanas)
- [ ] Optimización de performance
- [ ] Validación de seguridad completa
- [ ] Backup y recovery
- [ ] Documentación y training

---

## 🎯 **CRITERIOS DE ACEPTACIÓN**

### **Funcionales:**
- [ ] Médico puede iniciar/grabar conversación con un botón
- [ ] Sistema transcribe en tiempo real (feedback visual)
- [ ] Borrador se genera automáticamente post-consulta
- [ ] Interfaz permite edición fácil y rápida
- [ ] Informe final se integra automáticamente en historia clínica
- [ ] Búsqueda funciona en texto de transcripciones

### **No Funcionales:**
- [ ] Latencia < 2 segundos en transcripción
- [ ] 99.9% disponibilidad del servicio
- [ ] Cifrado end-to-end verificado
- [ ] Cumplimiento normativo documentado
- [ ] Backup automático cada 24h

---

## 📊 **MÉTRICAS DE ÉXITO**

### **Eficiencia:**
- **Reducción 70%** tiempo de documentación médica
- **Ahorro 5-7 horas** semanales por médico en papeleo
- **Incremento 40%** en completitud de historiales

### **Calidad:**
- **95% de precisión** en transcripción médica
- **Reducción 60%** en errores de documentación
- **Mejora 80%** en consistencia de formatos

### **Adopción:**
- **90% de médicos** usando el sistema en 30 días
- **Satisfacción >4.5/5** en encuestas de usuario
- **0 quejas** por pérdida de datos o seguridad

## 🖥️ **UI y Herramientas**
- **[Dash de recepción]** Timer visible con botones iniciar/pausa/finalizar y alertas cuando se superen umbrales de tiempo.
- **[Reporte de facturación]** Listado de cargos por emergencia mostrando horas, tarifas aplicadas y responsable de cierre.
- **[Auditoría]** Registro de quién modificó tiempos o tarifas.

## 🤖 **Automatización e IA (futuro)**
- **[Asistente]** Permitir que personal de emergencia use voz/comandos para iniciar o cerrar el conteo.
- **[Alertas inteligentes]** Notificar a supervisión si un caso excede límites (por ejemplo >4 horas sin alta).

## 🧪 **Implementación sugerida**
- **Sprint 1**: Migraciones `Ingreso` + modelo `EmergenciaTarifa` + endpoints para recepción.
- **Sprint 2**: UI recepción (timer) + cálculo backend de horas + pruebas.
- **Sprint 3**: Integración con facturación (líneas automáticas, reportes) + auditoría.
- **Sprint 4**: Integraciones opcionales (IA, alertas avanzadas, reporting).

---

# 💊 PROYECTO: CONTROL DE ENTREGAS Y DOSIS DE FARMACIA

## 🎯 **Objetivos Clave**
- **[Trazabilidad total]** Registrar quién solicita, autoriza y recibe cada medicamento (doctor, auxiliar, enfermera).
- **[Control de dosis]** Forzar captura de dosis administrada y remanente para presentaciones parciales (ampollas, viales).
- **[Prevención de desvíos]** Auditar cada entrega con firmas digitales/contraseñas y comparar contra facturación.

## 🧱 **Modelos y Relaciones**
- **[Nuevo `DispensacionMedicamento`]** (en `farmacia/`): campos `medicamento`, `presentacion`, `cantidad_entregada`, `cantidad_utilizada`, `cantidad_devuelta`, `dosis_unidad`, `unidad_medida`, `paciente`, `ingreso`, `solicitado_por`, `autorizado_por`, `entregado_por`, `recibido_por`, `estado`, `timestamp`.
- **[Bitácora de dosis]** `RegistroDosis` enlazado a `DispensacionMedicamento` para cada administración parcial con `dosis_ml`, `responsable`, `fecha_hora`, `ubicacion`.
- **[Integración inventario]** Descontar stock según `cantidad_entregada` pero obligar reconciliación con `cantidad_utilizada + cantidad_devuelta`.

## 🔄 **Flujo Operativo**
- **[Solicitud]** Enfermería solicita medicinas vía checklist; sistema identifica quién aprueba (doctor) y quién prepara (auxiliar/pharmacist).
- **[Entrega]** Farmacia registra `DispensacionMedicamento`, exige autenticación del receptor (PIN/credencial) y toma foto o firma opcional.
- **[Administración]** Enfermería registra dosis exacta (`RegistroDosis`) durante la atención (emergencia, hospitalización, UCI). Si queda remanente, el sistema obliga a devolverlo o justificar su consumo.
- **[Cierre]** Farmacia valida devoluciones y cambia estado a “completado”; diferencias generan alerta para supervisión y auditoría.

## 🖥️ **UI y Validaciones**
- **[Dashboard farmacia]** Lista de solicitudes pendientes, entregas en curso y alertas de dosis no justificadas.
- **[Formulario entrega]** Campos dinámicos según tipo de presentación (ml, tabletas, viales); pedir `dosis_ml` cuando corresponda.
- **[Alertas]** Si la dosis registrada supera lo entregado o no se devuelve remanente, se crea ticket para revisión.

## 🤖 **Automatización (IA opcional)**
- **[Asistente farmacia]** Reconoce comandos de voz para registrar entrega (“Registrar entrega de 1 ampolla de Irtopan para paciente X, dosis 300 ml”).
- **[Verificación inteligente]** IA compara patrones de consumo y detecta anomalías (por ejemplo, siempre se pierde 500 ml en turnos específicos).

## 🧪 **Roadmap propuesto**
- **Sprint 1**: Modelos `DispensacionMedicamento`, `RegistroDosis`, endpoints REST, reglas de validación básica.
- **Sprint 2**: UI de farmacia y enfermería (solicitud, entrega, administración) con autenticación de roles.
- **Sprint 3**: Integración inventario/facturación, reportes, alertas de divergencia.
- **Sprint 4**: IA asistente + analítica avanzada + documentación y capacitación.

---

# 📥 **FASE 0: PREPARACIÓN DEL ENTORNO**

### **1. CLONACIÓN Y CONFIGURACIÓN INICIAL**
```bash
# Timeline: Día 1
git clone [tu-repo-inca]
cd inca-system
python -m venv venv
source venv/bin/activate  # o venv\Scripts\activate en Windows
```

### **2. INSTALACIÓN DE DEPENDENCIAS**
```bash
# Timeline: Día 1
pip install -r requirements.txt
pip install vocode==0.19.20
pip install python-dotenv
```

### **3. CONFIGURACIÓN DE VARIABLES DE ENTORNO**
```bash
# Timeline: Día 1
# Crear archivo .env en root del proyecto
VOCODE_API_KEY=tu_vocode_key
OPENAI_API_KEY=tu_openai_key
DEEPGRAM_API_KEY=tu_deepgram_key
ELEVENLABS_API_KEY=tu_elevenlabs_key

# Configuración de base de datos
DB_HOST=localhost
DB_NAME=inca_medical
DB_USER=usuario
DB_PASS=contraseña
```

---

## 🗂️ **FASE 1: ESTRUCTURA DE ARCHIVOS**

### **1. CREACIÓN DEL MÓDULO DE TRANSCRIPCIÓN**
```bash
# Timeline: Día 2
mkdir transcripcion_voz
cd transcripcion_voz

# Estructura de archivos a crear
touch __init__.py
touch models.py
touch views.py
touch forms.py
touch services.py
touch utils.py
touch urls.py
touch signals.py
```

### **2. CONFIGURACIÓN DJANGO**
```python
# En settings.py - agregar a INSTALLED_APPS
# Timeline: Día 2
INSTALLED_APPS = [
    # apps existentes...
    'transcripcion_voz',
]
```

---

## 🗃️ **FASE 2: MODELOS DE DATOS**

### **1. DISEÑO DE MODELOS** (Día 3)
```python
# transcripcion_voz/models.py
# Modelos a crear:

# TranscripcionConsulta
# - Campos: paciente(FK), medico(FK), estado, audio_original, texto_transcrito

# InformeMedico
# - Campos: transcripcion(OneToOne), borrador_ia, informe_final, editado_por

# ConfigAudio
# - Campos: usuario(FK), dispositivo_audio, calidad, idioma
```

### **2. MIGRACIONES** (Día 3)
```bash
python manage.py makemigrations transcripcion_voz
python manage.py migrate transcripcion_voz
```

---

## 🔧 **FASE 3: SERVICIOS VOCODE**

### **1. CONFIGURACIÓN SERVICIO PRINCIPAL** (Día 4-5)
```python
# transcripcion_voz/services.py
# Servicios a implementar:

# AudioCaptureService
# - Métodos: iniciar_grabacion(), detener_grabacion(), procesar_audio()

# TranscriptionService  
# - Métodos: transcribir_audio(), validar_calidad()

# InformeGenerationService
# - Métodos: generar_borrador(), estructurar_informe()
```

### **2. INTEGRACIÓN CON PROVIDERS** (Día 6)
```python
# Configuración de proveedores:
# - Deepgram: STT (Speech-to-Text)
# - OpenAI: LLM para estructuración
# - ElevenLabs: TTS (Text-to-Speech) - opcional
```

---

## 🌐 **FASE 4: VISTAS Y URLs**

### **1. RUTAS PRINCIPALES** (Día 7)
```python
# transcripcion_voz/urls.py
# Endpoints a crear:

# /transcripcion/iniciar/ - POST
# /transcripcion/detener/ - POST  
# /transcripcion/editar/<id>/ - GET/POST
# /transcripcion/aprobar/<id>/ - POST
# /transcripcion/historial/ - GET
```

### **2. VISTAS PRINCIPALES** (Día 8-9)
```python
# transcripcion_voz/views.py
# Vistas a implementar:

# IniciarTranscripcionView
# - Lógica: validar permisos, iniciar grabación

# EditarTranscripcionView  
# - Lógica: cargar borrador, permitir edición

# AprobarTranscripcionView
# - Lógica: guardar en historia clínica, notificar
```

---

## 🎨 **FASE 5: INTERFACES DE USUARIO**

### **1. TEMPLATES** (Día 10-11)
```bash
# Estructura de templates
templates/transcripcion_voz/
    ├── iniciar_transcripcion.html
    ├── editar_informe.html
    ├── historial_transcripciones.html
    └── componentes/
        ├── audio_controls.html
        └── editor_informe.html
```

### **2. COMPONENTES FRONTEND** (Día 12)
```javascript
// Archivos static/ a crear:
// js/audio-recorder.js - Grabación de audio
// js/real-time-transcription.js - Transcripción en tiempo real  
// js/informe-editor.js - Editor de informes
```

---

## 🔗 **FASE 6: INTEGRACIÓN CON MÓDULOS EXISTENTES**

### **1. CON HISTORIA CLÍNICA** (Día 13)
```python
# En historia_clinica/views.py
# Modificar creación de consultas para incluir opción de transcripción

# En historia_clinica/models.py
# Agregar relación con TranscripcionConsulta
```

### **2. SEÑALES Y EVENTOS** (Día 14)
```python
# transcripcion_voz/signals.py
# Señales a implementar:

# post_save Transcipcion -> crear entrada historia clínica
# pre_delete Transcripcion -> limpiar archivos de audio
```

---

## 🧪 **FASE 7: TESTING**

### **1. PRUEBAS UNITARIAS** (Día 15-16)
```bash
# Crear archivos de test
touch tests/test_models.py
touch tests/test_services.py  
touch tests/test_views.py
touch tests/test_integration.py
```

### **2. EJECUCIÓN DE TESTS** (Día 17)
```bash
python manage.py test transcripcion_voz --verbosity=2
```

---

## 🔒 **FASE 8: SEGURIDAD Y OPTIMIZACIÓN**

### **1. CONFIGURACIÓN SEGURIDAD** (Día 18)
```python
# En settings.py
# Configurar: CORS, CSRF, permisos de archivos de audio
# Implementar: cifrado de archivos, autenticación JWT para APIs
```

### **2. OPTIMIZACIÓN** (Día 19)
```python
# Implementar:
# - Cache para transcripciones frecuentes
# - Background tasks para procesamiento pesado
# - Compresión de archivos de audio
```

---

## 🚀 **FASE 9: DEPLOYMENT**

### **1. PREPARACIÓN PRODUCCIÓN** (Día 20)
```bash
# Configurar:
# - Gunicorn/Uvicorn para ASGI
# - Nginx para servir archivos estáticos
# - Redis para cache
# - Celery para tareas background
```

### **2. VARIABLES PRODUCCIÓN** (Día 21)
```bash
# Configurar en servidor:
# - Variables de entorno seguras
# - Certificados SSL
# - Backup automático
# - Monitorización
```

---

## 📋 **FASE 10: DOCUMENTACIÓN Y TRAINING**

### **1. DOCUMENTACIÓN TÉCNICA** (Día 22)
```bash
# Crear documentación:
docs/
├── instalacion.md
├── configuracion_vocode.md
├── api_endpoints.md
└── troubleshooting.md
```

### **2. MANUAL USUARIO** (Día 23)
```bash
# Crear guías de usuario:
manuales/
├── medico_transcripcion.md
├── administrador_config.md
└── preguntas_frecuentes.md
```

---

## 🔄 **CRONOGRAMA DETALLADO**

### **SEMANA 1: FUNDACIÓN**
- **Día 1-2**: Entorno y estructura
- **Día 3-4**: Modelos y base de datos
- **Día 5**: Servicios core Vocode

### **SEMANA 2: DESARROLLO**
- **Día 6-7**: APIs y vistas
- **Día 8-9**: Frontend y UI
- **Día 10**: Integración módulos

### **SEMANA 3: CALIDAD**
- **Día 11-12**: Testing y bugs
- **Día 13-14**: Seguridad y optimización

### **SEMANA 4: PRODUCCIÓN**
- **Día 15-16**: Deployment
- **Día 17**: Documentación
- **Día 18-19**: Training y lanzamiento

---

## ⚠️ **PUNTOS CRÍTICOS DE ATENCIÓN**

### **TÉCNICOS:**
- **Configuración APIs externas** - Día 6 (crítico)
- **Integración con historia clínica** - Día 13 (crítico)  
- **Manejo de archivos de audio** - Día 4-5 (complejo)

### **SEGURIDAD:**
- **Cifrado de datos médicos** - Día 18 (obligatorio)
- **Permisos y roles** - Día 14 (obligatorio)
- **Backup y recovery** - Día 21 (obligatorio)

### **PERFORMANCE:**
- **Procesamiento en background** - Día 19 (recomendado)
- **Cache de transcripciones** - Día 19 (recomendado)
- **Optimización de audio** - Día 19 (recomendado)

---
############################################################
implementar  nueva bd en la nube el manual esta dentro de el archivo llamado #turso-manual.md

Key= eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJKUzNsT0tiZUVmQ291b2FIS2ROVi1nIn0.40hk9EUBwGro4bqfZyUYAojsV6Yu_Ghp_u1sq-oniCsJfohb4knR2Yrdl1estehHJ8H6GWkFs7qP-4Tn7gDLAA
URL= libsql://yari-medicdb-drcoa.aws-us-east-1.turso.io

# 🎨 GUÍA DE MIGRACIÓN UI - SOFTMEDIC

## 📋 Resumen de Cambios

Se ha implementado un **sistema de diseño moderno y atractivo** para mejorar significativamente la experiencia de usuario (UX) del sistema Yari Medic, especialmente para médicos y secretarias.

## ✨ Nuevas Características

### 🎨 Sistema de Diseño
- **Design System completo** con variables CSS consistentes
- **Colores modernos** con gradientes atractivos
- **Tipografía mejorada** con mejor legibilidad
- **Espaciado consistente** para mejor organización visual

### 🖱️ Componentes Modernos
- **Botones modernos** con efectos hover y transiciones suaves
- **Cards atractivas** con sombras y bordes redondeados
- **Formularios mejorados** con validación visual en tiempo real
- **Tablas modernas** con mejor organización y legibilidad
- **Alertas visuales** con iconos y colores distintivos

### 📱 Experiencia de Usuario
- **Sidebar moderno** con navegación intuitiva
- **Header mejorado** con información contextual
- **Dashboard interactivo** con métricas visuales
- **Modo oscuro/claro** con transiciones suaves
- **Responsive design** optimizado para móviles

### ⚡ Funcionalidades Avanzadas
- **Animaciones suaves** para mejor feedback visual
- **Notificaciones toast** para mejor comunicación
- **Validación en tiempo real** en formularios
- **Búsqueda mejorada** con filtros visuales
- **Acciones rápidas** con iconos intuitivos

## 🚀 Instrucciones de Migración

### 1. Preparación
```bash
# Hacer backup de la base de datos
python manage.py dumpdata > backup_before_ui_migration.json

# Crear backup de templates existentes
python backup_templates.py
```

### 2. Aplicar Cambios
```bash
# Ejecutar el script de migración
python update_ui_templates.py

# Recopilar archivos estáticos
python manage.py collectstatic

# Reiniciar el servidor
python manage.py runserver
```

### 3. Verificación
- ✅ Verificar que el dashboard se vea moderno
- ✅ Probar la navegación del sidebar
- ✅ Verificar formularios con validación
- ✅ Probar modo oscuro/claro
- ✅ Verificar responsive en móviles

## 📁 Archivos Modificados

### CSS
- `static/css/design-system.css` - Sistema de diseño completo

### Templates Base
- `dashboard/templates/base.html` - Template base modernizado
- `dashboard/templates/base_modern.html` - Template base nuevo

### Templates Dashboard
- `dashboard/templates/dashboard/index.html` - Dashboard moderno
- `dashboard/templates/dashboard/index_modern.html` - Dashboard nuevo

### Templates Pacientes
- `pacientes/templates/pacientes/lista_pacientes.html` - Lista moderna
- `pacientes/templates/pacientes/crear_paciente.html` - Formulario moderno

## 🔧 Personalización

### Variables CSS
Puedes personalizar el diseño modificando las variables en `design-system.css`:

```css
:root {
  --primary-color: #2563eb;    /* Color principal */
  --success-color: #10b981;    /* Color de éxito */
  --warning-color: #f59e0b;    /* Color de advertencia */
  --error-color: #ef4444;      /* Color de error */
  /* ... más variables */
}
```

### Componentes
Todos los componentes están documentados en el CSS con ejemplos de uso.

## 🐛 Solución de Problemas

### Problema: Estilos no se aplican
**Solución:**
```bash
python manage.py collectstatic --clear
python manage.py collectstatic
```

### Problema: Template no se actualiza
**Solución:**
```bash
# Verificar que el template moderno existe
ls -la templates/*/template_modern.html

# Ejecutar script de actualización
python update_ui_templates.py
```

### Problema: JavaScript no funciona
**Solución:**
- Verificar que jQuery esté cargado
- Verificar que Bootstrap JS esté cargado
- Revisar la consola del navegador

## 📞 Soporte

Si encuentras algún problema:
1. Revisar los logs del servidor Django
2. Verificar la consola del navegador
3. Restaurar desde backup si es necesario:
   ```bash
   cp templates/*/template.html.backup templates/*/template.html
   ```

## 🎯 Próximos Pasos

1. **Aplicar a más módulos**: Extender el diseño a citas, médicos, facturación, etc.
2. **Optimizar performance**: Implementar lazy loading y optimizaciones
3. **Añadir más animaciones**: Mejorar la experiencia con micro-interacciones
4. **Testing**: Crear tests para los nuevos componentes
5. **Documentación**: Crear guía de componentes para desarrolladores

---

**¡El sistema Yari Medic ahora tiene una interfaz moderna, atractiva y fácil de usar! 🎉**

# Turso Database Manual

Welcome to Turso database manual!

## Table of contents

- [Turso Database Manual](#turso-database-manual)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Getting Started](#getting-started)
    - [Limitations](#limitations)
  - [Transactions](#transactions)
    - [Deferred transaction lifecycle](#deferred-transaction-lifecycle)
  - [The SQL shell](#the-sql-shell)
    - [Shell commands](#shell-commands)
    - [Command line options](#command-line-options)
  - [The SQL language](#the-sql-language)
    - [`ALTER TABLE` — change table definition](#alter-table--change-table-definition)
    - [`BEGIN TRANSACTION` — start a transaction](#begin-transaction--start-a-transaction)
    - [`COMMIT TRANSACTION` — commit the current transaction](#commit-transaction--commit-the-current-transaction)
    - [`CREATE INDEX` — define a new index](#create-index--define-a-new-index)
    - [`CREATE TABLE` — define a new table](#create-table--define-a-new-table)
    - [`DELETE` - delete rows from a table](#delete---delete-rows-from-a-table)
    - [`DROP INDEX` - remove an index](#drop-index---remove-an-index)
    - [`DROP TABLE` — remove a table](#drop-table--remove-a-table)
    - [`END TRANSACTION` — commit the current transaction](#end-transaction--commit-the-current-transaction)
    - [`INSERT` — create new rows in a table](#insert--create-new-rows-in-a-table)
    - [`ROLLBACK TRANSACTION` — abort the current transaction](#rollback-transaction--abort-the-current-transaction)
    - [`SELECT` — retrieve rows from a table](#select--retrieve-rows-from-a-table)
    - [`UPDATE` — update rows of a table](#update--update-rows-of-a-table)
  - [JavaScript API](#javascript-api)
    - [Installation](#installation)
    - [Getting Started](#getting-started-1)
  - [SQLite C API](#sqlite-c-api)
    - [Basic operations](#basic-operations)
      - [`sqlite3_open`](#sqlite3_open)
      - [`sqlite3_prepare`](#sqlite3_prepare)
      - [`sqlite3_step`](#sqlite3_step)
      - [`sqlite3_column`](#sqlite3_column)
    - [WAL manipulation](#wal-manipulation)
      - [`libsql_wal_frame_count`](#libsql_wal_frame_count)
  - [Encryption](#encryption)
  - [CDC](#cdc-early-preview)
  - [Appendix A: Turso Internals](#appendix-a-turso-internals)
    - [Frontend](#frontend)
      - [Parser](#parser)
      - [Code generator](#code-generator)
      - [Query optimizer](#query-optimizer)
    - [Virtual Machine](#virtual-machine)
    - [MVCC](#mvcc)
    - [Pager](#pager)
    - [I/O](#io)
    - [Encryption](#encryption-1)
    - [References](#references)

## Introduction

Turso is an in-process relational database engine, aiming towards full compatibility with SQLite.

Unlike client-server database systems such as PostgreSQL or MySQL, which require applications to communicate over network protocols for SQL execution,
an in-process database is in your application memory space.
This embedded architecture eliminates network communication overhead, allowing for the best case of low read and write latencies in the order of sub-microseconds.

### Getting Started

You can install Turso on your computer as follows:

```
curl --proto '=https' --tlsv1.2 -LsSf \
  https://github.com/tursodatabase/turso/releases/latest/download/turso_cli-installer.sh | sh
```


```
brew install turso
```

When you have the software installed, you can start a SQL shell as follows:

```console
$ tursodb
Turso
Enter ".help" for usage hints.
Connected to a transient in-memory database.
Use ".open FILENAME" to reopen on a persistent database
turso> SELECT 'hello, world';
hello, world
```

### Limitations

Turso aims towards full SQLite compatibility but has the following limitations:

* Query result ordering is not guaranteed to be the same (see [#2964](https://github.com/tursodatabase/turso/issues/2964) for more discussion)
* No multi-process access
* No multi-threading
* No savepoints
* No triggers
* No views
* No vacuum
* UTF-8 is the only supported character encoding

For more detailed list of SQLite compatibility, please refer to [COMPAT.md](../COMPAT.md).

#### MVCC limitations

The MVCC implementation is experimental and has the following limitations:

* Indexes cannot be created and databases with indexes cannot be used.
* All the data is eagerly loaded from disk to memory on first access so using big databases may take a long time to start, and will consume a lot of memory
* Only `PRAGMA wal_checkpoint(TRUNCATE)` is supported and it blocks both readers and writers
* Many features may not work, work incorrectly, and/or cause a panic.
* Queries may return incorrect results
* If a database is written to using MVCC and then opened again without MVCC, the changes are not visible unless first checkpointed

## The SQL shell

The `tursodb` command provides an interactive SQL shell, similar to `sqlite3`. You can start it in in-memory mode as follows:

```console
$ tursodb
Turso
Enter ".help" for usage hints.
Connected to a transient in-memory database.
Use ".open FILENAME" to reopen on a persistent database
turso> SELECT 'hello, world';
hello, world
```

### Shell commands

The shell supports commands in addition to SQL statements. The commands start with a dot (".") followed by the command. The supported commands are:

| Command | Description |
|---------|-------------|
| `.schema` | Display the database schema |
| `.dump` | Dump database contents as SQL statements |

### Command line options

The SQL shell supports the following command line options:

| Option | Description |
|--------|-------------|
| `-m`, `--output-mode` `<mode>` | Configure output mode. Supported values for `<mode>`: <ul><li>`pretty` for pretty output (default)</li><li>`list` for minimal SQLite compatible format</li></ul>
| `-q`, `--quiet` | Don't display program information at startup |
| `-e`, `--echo` | Print commands before execution |
| `--readonly` | Open database in read-only mode |
| `-h`, `--help` | Print help |
| `-V`, `--version` | Print version |
| `--mcp` | Start a MCP server instead of the interactive shell |
| `--experimental-encryption` | Enable experimental encryption at rest feature. **Note:** the feature is not production ready so do not use it for critical data right now. |
| `--experimental-mvcc` | Enable experimental MVCC feature. **Note:** the feature is not production ready so do not use it for critical data right now. |
| `--experimental-strict` | Enable experimental strict schema feature. **Note**: the feature is not production ready so do not use it for critical data right now. |
| `--experimental-views` | Enable experimental views feature. **Note**: the feature is not production ready so do not use it for critical data right now. |

## Transactions

A transaction is a sequence of one or more SQL statements that execute as a single, atomic unit of work.
A transaction ensures **atomicity** and **isolation**, meaning that either all SQL statements are executed or none of them are, and that concurrent transactions don't interfere with other transactions.
Transactions maintain database integrity in the presence of errors, crashes, and concurrent access.

Turso supports three types of transactions: **deferred**, **immediate**, and **concurrent** transactions:

* **Deferred (default)**: The transaction begins in a suspended state and does not acquire locks immediately. It starts a read transaction when the first read SQL statement (e.g., `SELECT`) runs, and upgrades to a write transaction only when the first write SQL statement (e.g., `INSERT`, `UPDATE`, `DELETE`) executes. This mode allows concurrency for reads and delays write locks, which can reduce contention.
* **Immediate**: The transaction starts immediately with a reserved write lock, preventing other write transactions from starting concurrently but allowing reads. It attempts to acquire the write lock right away on the `BEGIN` statement, which can fail if another write transaction exists. The `EXCLUSIVE` mode is always an alias for `IMMEDIATE` in Turso, just like it is in SQLite in WAL mode.
* **Concurrent (MVCC only)**: The transaction begins immediately and allows multiple concurrent read and write transactions. When a concurrent transaction commits, the database performs row-level conflict detection and returns a `SQLITE_BUSY` (write-write conflict) error if the transaction attempted to modify a row that was concurrently modified by another transaction. This mode provides the highest level of concurrency at the cost of potential transaction conflicts that must be retried by the application. The transaction isolation level provided by concurrent transactions is snapshot isolation.

### Deferred transaction lifecycle

When the `BEGIN DEFERRED TRANSACTION` statement executes, the database acquires no snapshot or locks. Instead, the transaction is in a suspended state until the first read or write SQL statement executes. When the first read statement executes, a read transaction begins. The database allows multiple read transactions to exist concurrently. When the first write statement executes, a read transaction is either upgraded to a write transaction or a write transaction begins. The database allows a single write transaction at a time. Concurrent write transactions fail with `SQLITE_BUSY` error.

If a deferred transaction remains unused (no reads or writes), it is automatically restarted by the database if another write transaction commits before the transaction is used. However, if the deferred transaction has already performed reads and another concurrent write transaction commits, it cannot automatically restart due to potential snapshot inconsistency. In this case, the deferred transaction must be manually rolled back and restarted by the application.

### Concurrent transaction lifecycle

Concurrent transactions are only available when MVCC (Multi-Version Concurrency Control) is enabled in the database. They use optimistic concurrency control to allow multiple transactions to modify the database simultaneously.

When the `BEGIN CONCURRENT TRANSACTION` statement executes, the database:

1. Assigns a unique transaction ID to the transaction
2. Records a begin timestamp from the logical clock
3. Creates an empty read set and write set to track accessed rows
4. Does **not** acquire any locks

Unlike deferred transactions which delay locking, concurrent transactions never acquire locks. Instead, they rely on MVCC's snapshot isolation and conflict detection at commit time.

#### Read snapshot isolation

Each concurrent transaction reads from a consistent snapshot of the database as of its begin timestamp. This means:

- Reads see all data committed before the transaction's begin timestamp
- Reads do **not** see writes from other transactions that commit after this transaction starts
- Reads from the same transaction are consistent (repeatable reads within the transaction)
- Multiple concurrent transactions can read and write simultaneously without blocking each other

All rows read by the transaction are tracked in the transaction's read set, and all rows written are tracked in the write set.

#### Commit and conflict detection

When a concurrent transaction commits, the database performs these steps:

1. **Exclusive transaction check**: If there is an active exclusive transaction (started with `BEGIN IMMEDIATE` or a `BEGIN DEFERRED` that upgraded to a write transaction), the concurrent transaction **cannot commit** and receives a `SQLITE_BUSY` error. Concurrent transactions can read and write concurrently with exclusive transactions, but cannot commit until the exclusive transaction completes.

2. **Write-write conflict detection**: For each row in the transaction's write set, the database checks if the row was modified by another transaction. A write-write conflict occurs when:
   - The row is currently being modified by another active transaction, or
   - The row was modified by a transaction that committed after this transaction's begin timestamp

3. **Commit or abort**: If no conflicts are detected, the transaction commits successfully. All row versions in the write set have their begin timestamps updated to the transaction's commit timestamp, making them visible to future transactions. If a conflict is detected, the transaction fails with a `SQLITE_BUSY` error and must be rolled back and retried by the application.

#### Interaction with exclusive transactions

Concurrent transactions can coexist with exclusive transactions (deferred and immediate), but with important restrictions:

- **Concurrent transactions can read and write** while an exclusive transaction is active
- **Concurrent transactions cannot commit** while an exclusive transaction holds the exclusive lock
- **Exclusive transactions block concurrent transaction commits**, not their reads or writes

This design allows concurrent transactions to make progress during an exclusive transaction, but ensures that exclusive transactions truly have exclusive write access when needed (for example, schema changes).

**Best practice**: For maximum concurrency in MVCC mode, use `BEGIN CONCURRENT` for all write transactions. Only use `BEGIN IMMEDIATE` or `BEGIN DEFERRED` when you need exclusive write access that prevents any concurrent commits.

## The SQL language

### `ALTER TABLE` — change table definition

**Synopsis:**

```sql
ALTER TABLE old_name RENAME TO new_name

ALTER TABLE table_name ADD COLUMN column_name [ column_type ]

ALTER TABLE table_name DROP COLUMN column_name
```

**Example:**

```console
turso> CREATE TABLE t(x);
turso> .schema t;
CREATE TABLE t (x);
turso> ALTER TABLE t ADD COLUMN y TEXT;
turso> .schema t
CREATE TABLE t ( x , y TEXT );
turso> ALTER TABLE t DROP COLUMN y;
turso> .schema t
CREATE TABLE t ( x  );
```

### `BEGIN TRANSACTION` — start a transaction

**Synopsis:**

```sql
BEGIN [ transaction_mode ] [ TRANSACTION ]
```

where `transaction_mode` is one of the following:

* A `DEFERRED` transaction in a suspended state and does not acquire locks immediately. It starts a read transaction when the first read SQL statement (e.g., `SELECT`) runs, and upgrades to a write transaction only when the first write SQL statement (e.g., `INSERT`, `UPDATE`, `DELETE`) executes.
* An `IMMEDIATE` transaction starts immediately with a reserved write lock, preventing other write transactions from starting concurrently but allowing reads. It attempts to acquire the write lock right away on the `BEGIN` statement, which can fail if another write transaction exists.
* An `EXCLUSIVE` transaction is always an alias for `IMMEDIATE`.
* A `CONCURRENT` transaction begins immediately, but allows other concurrent transactions.

**See also:**

* [Transactions](#transactions)
* [END TRANSACTION](#end-transaction--commit-the-current-transaction)

### `COMMIT TRANSACTION` — commit the current transaction

**Synopsis:**

```sql
COMMIT [ TRANSACTION ]
```

**See also:**

* [END TRANSACTION](#end-transaction--commit-the-current-transaction)

### `CREATE INDEX` — define a new index

> [!NOTE]  
> Indexes are currently experimental in Turso and not enabled by default.

**Synopsis:**

```sql
CREATE INDEX [ index_name ] ON table_name ( column_name )
```

**Example:**

```
turso> CREATE TABLE t(x);
turso> CREATE INDEX t_idx ON t(x);
```

### `CREATE TABLE` — define a new table

**Synopsis:**

```sql
CREATE TABLE table_name ( column_name [ column_type ], ... )
```

**Example:**

```console
turso> DROP TABLE t;
turso> CREATE TABLE t(x);
turso> .schema t
CREATE TABLE t (x);
```

### `DELETE` - delete rows from a table

**Synopsis:**

```sql
DELETE FROM table_name [ WHERE expression ]
```

**Example:**

```console
turso> DELETE FROM t WHERE x > 1;
```

### `DROP INDEX` - remove an index

> [!NOTE]  
> Indexes are currently experimental in Turso and not enabled by default.

**Example:**

```console
turso> DROP INDEX idx;
```

### `DROP TABLE` — remove a table

**Example:**

```console
turso> DROP TABLE t;
```

### `END TRANSACTION` — commit the current transaction

```sql
END [ TRANSACTION ]
```

**See also:**

* `COMMIT TRANSACTION`

### `INSERT` — create new rows in a table

**Synopsis:**

```sql
INSERT INTO table_name [ ( column_name, ... ) ] VALUES ( value, ... ) [, ( value, ... ) ...]
```

**Example:**

```
turso> INSERT INTO t VALUES (1), (2), (3);
turso> SELECT * FROM t;
┌───┐
│ x │
├───┤
│ 1 │
├───┤
│ 2 │
├───┤
│ 3 │
└───┘
```

### `ROLLBACK TRANSACTION` — abort the current transaction

```sql
ROLLBACK [ TRANSACTION ]
```

### `SELECT` — retrieve rows from a table

**Synopsis:**

```sql
SELECT expression
    [ FROM table-or-subquery ]
    [ WHERE condition ]
    [ GROU BY expression ]
```

**Example:**

```console
turso> SELECT 1;
┌───┐
│ 1 │
├───┤
│ 1 │
└───┘
turso> CREATE TABLE t(x);
turso> INSERT INTO t VALUES (1), (2), (3);
turso> SELECT * FROM t WHERE x >= 2;
┌───┐
│ x │
├───┤
│ 2 │
├───┤
│ 3 │
└───┘
```

### `UPDATE` — update rows of a table

**Synopsis:**

```sql
UPDATE table_name SET column_name = value [WHERE expression]
```

**Example:**

```console
turso> CREATE TABLE t(x);
turso> INSERT INTO t VALUES (1), (2), (3);
turso> SELECT * FROM t;
┌───┐
│ x │
├───┤
│ 1 │
├───┤
│ 2 │
├───┤
│ 3 │
└───┘
turso> UPDATE t SET x = 4 WHERE x >= 2;
turso> SELECT * FROM t;
┌───┐
│ x │
├───┤
│ 1 │
├───┤
│ 4 │
├───┤
│ 4 │
└───┘
```

## JavaScript API

Turso supports a JavaScript API, both with native and WebAssembly package options.

Please read the [JavaScript API reference](docs/javascript-api-reference.md) for more information.

### Installation

Installing the native package:

```console
npm i @tursodatabase/database
```

Installing the WebAssembly package:

```console
npm i @tursodatabase/database --cpu wasm32
```

### Getting Started

To use Turso from JavaScript application, you need to import `Database` type from the `@tursodatabase/database` package.
You can the prepare a statement with `Database.prepare` method and execute the SQL statement with `Statement.get()` method.

```
import { connect } from '@tursodatabase/database';

const db = await connect('turso.db');
const row = db.prepare('SELECT 1').get();
console.log(row);
```

## SQLite C API

Turso supports a subset of the SQLite C API, including libSQL extensions.

### Basic operations

#### `sqlite3_open` 

Open a connection to a database.

**Synopsis:**

```c
int sqlite3_open(const char *filename, sqlite3 **db_out);
int sqlite3_open_v2(const char *filename, sqlite3 **db_out, int _flags, const char *_z_vfs);
```

#### `sqlite3_prepare`

Prepare a SQL statement for execution.

**Synopsis:**

```c
int sqlite3_prepare_v2(sqlite3 *db, const char *sql, int _len, sqlite3_stmt **out_stmt, const char **_tail);
```

#### `sqlite3_step`

Evaluate a prepared statement until it yields the next row or completes.

**Synopsis:**

```c
int sqlite3_step(sqlite3_stmt *stmt);
```

#### `sqlite3_column`

Return the value of a column for the current row of a statement.

**Synopsis:**

```c
int sqlite3_column_type(sqlite3_stmt *_stmt, int _idx);
int sqlite3_column_count(sqlite3_stmt *_stmt);
const char *sqlite3_column_decltype(sqlite3_stmt *_stmt, int _idx);
const char *sqlite3_column_name(sqlite3_stmt *_stmt, int _idx);
int64_t sqlite3_column_int64(sqlite3_stmt *_stmt, int _idx);
double sqlite3_column_double(sqlite3_stmt *_stmt, int _idx);
const void *sqlite3_column_blob(sqlite3_stmt *_stmt, int _idx);
int sqlite3_column_bytes(sqlite3_stmt *_stmt, int _idx);
const unsigned char *sqlite3_column_text(sqlite3_stmt *stmt, int idx);
```

### WAL manipulation

#### `libsql_wal_frame_count`

Get the number of frames in the WAL.

**Synopsis:**

```c
int libsql_wal_frame_count(sqlite3 *db, uint32_t *p_frame_count);
```

**Description:**

The `libsql_wal_frame_count` function returns the number of frames in the WAL
in the `p_frame_count` parameter.

**Return Values:**

* `SQLITE_OK` if the number of frames in the WAL file is successfully returned.
* `SQLITE_MISUSE` if the `db` is NULL.
* SQLITE_ERROR if an error occurs while getting the number of frames in the WAL
  file.

**Safety Requirements:**

* The `db` parameter must be a valid pointer to a `sqlite3` database
  connection.
* The `p_frame_count` must be a valid pointer to a `u32` that will store the
* number of frames in the WAL file.

## Encryption

The work-in-progress RFC is [here](https://github.com/tursodatabase/turso/issues/2447).
To use encryption, you need to enable it via flag `experimental-encryption`.
To get started, generate a secure 32 byte key in hex: 

```shell
$ openssl rand -hex 32
2d7a30108d3eb3e45c90a732041fe54778bdcf707c76749fab7da335d1b39c1d
```

Specify the key and cipher at the time of db creation to use encryption. Here is [sample test](https://github.com/tursodatabase/turso/blob/main/tests/integration/query_processing/encryption.rs):

```shell
$ cargo run -- --experimental-encryption database.db

PRAGMA cipher = 'aegis256'; -- or 'aes256gcm'
PRAGMA hexkey = '2d7a30108d3eb3e45c90a732041fe54778bdcf707c76749fab7da335d1b39c1d';
```
Alternatively you can provide the encryption parameters directly in a **URI**. For example:
```shell
$ cargo run -- --experimental-encryption \
"file:database.db?cipher=aegis256&hexkey=2d7a30108d3eb3e45c90a732041fe54778bdcf707c76749fab7da335d1b39c1d"
```


> **Note:**  To reopen an already *encrypted database*, the file **must** be opened in URI format with the `cipher` and `hexkey` passed as URI parameters. Now, to reopen `database.db` the command below must be run:

```shell
$ cargo run -- --experimental-encryption \
   "file:database.db?cipher=aegis256hexkey=2d7a30108d3eb3e45c90a732041fe54778bdcf707c76749fab7da335d1b39c1d"
```


## CDC (Early Preview)

Turso supports [Change Data Capture](https://en.wikipedia.org/wiki/Change_data_capture), a powerful pattern for tracking and recording changes to your database in real-time. Instead of periodically scanning tables to find what changed, CDC automatically logs every insert, update, and delete as it happens per connection.

### Enabling CDC

```sql
PRAGMA unstable_capture_data_changes_conn('<mode>[,custom_cdc_table]');
```

### Parameters
- `<mode>` can be:
    - `off`: Turn off CDC for the connection
    - `id`: Logs only the `rowid` (most compact)
    - `before`: Captures row state before updates and deletes
    - `after`: Captures row state after inserts and updates
    - `full`: Captures both before and after states (recommended for complete audit trail)

- `custom_cdc` is optional, It lets you specify a custom table to capture changes.
If no table is provided, Turso uses a default `turso_cdc` table.


When **Change Data Capture (CDC)** is enabled for a connection, Turso automatically logs all modifications from that connection into a dedicated table (default: `turso_cdc`). This table records each change with details about the operation, the affected row or schema object, and its state **before** and **after** the modification.

> **Note:** Currently, the CDC table is a regular table stored explicitly on disk. If you use full CDC mode and update rows frequently, each update of size N bytes will be written three times to disk (once for the before state, once for the after state, and once for the actual value in the WAL). Frequent updates in full mode can therefore significantly increase disk I/O.



- **`change_id` (INTEGER)**  
  A monotonically increasing integer uniquely identifying each change record.(guaranteed by turso-db) 
  - Always strictly increasing.  
  - Serves as the primary key.  

- **`change_time` (INTEGER)**  
> turso-db guarantee nothing about properties of the change_time sequence 
  Local timestamp (Unix epoch, seconds) when the change was recorded.  
  - Not guaranteed to be strictly increasing (can drift or repeat).  

- **`change_type` (INTEGER)**  
  Indicates the type of operation:  
  - `1` → INSERT  
  - `0` → UPDATE (also used for ALTER TABLE)  
  - `-1` → DELETE (also covers DROP TABLE, DROP INDEX)  

- **`table_name` (TEXT)**  
  Name of the affected table.  
  - For schema changes (DDL), this is always `"sqlite_schema"`.  

- **`id` (INTEGER)**  
  Rowid of the affected row in the source table.  
  - For DDL operations: rowid of the `sqlite_schema` entry.  
  - **Note:** `WITHOUT ROWID` tables are not supported in the tursodb and CDC

- **`before` (BLOB)**  
  Full state of the row/schema **before** an UPDATE or DELETE
  - NULL for INSERT.  
  - For DDL changes, may contain the definition of the object before modification.  

- **`after` (BLOB)**  
  Full state of the row/schema **after** an INSERT or UPDATE
  - NULL for DELETE.  
  - For DDL changes, may contain the definition of the object after modification.  

- **`updates` (BLOB)**  
  Granular details about the change.  
  - For UPDATE: shows specific column modifications.  


> CDC records are visible even before a transaction commits. 
> Operations that fail (e.g., constraint violations) are not recorded in CDC.

> Changes to the CDC table itself are also logged to CDC table. if CDC is enabled for that connection.

```zsh
Example:
turso> PRAGMA unstable_capture_data_changes_conn('full');
turso> .tables
turso_cdc
turso> CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT
);
turso> INSERT INTO users VALUES (1, 'John'), (2, 'Jane');

UPDATE users SET name='John Doe' WHERE id=1;

DELETE FROM users WHERE id=2;

SELECT * FROM turso_cdc;
┌───────────┬─────────────┬─────────────┬───────────────┬────┬──────────┬──────────────────────────────────────────────────────────────────────────────┬───────────────┐
│ change_id │ change_time │ change_type │ table_name    │ id │ before   │ after                                                                        │ updates       │
├───────────┼─────────────┼─────────────┼───────────────┼────┼──────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────┤
│         1 │  1756713161 │           1 │ sqlite_schema │  2 │          │ ytableusersusersCREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT) │               │
├───────────┼─────────────┼─────────────┼───────────────┼────┼──────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────┤
│         2 │  1756713176 │           1 │ users         │  1 │          │       John                                                                      │               │
├───────────┼─────────────┼─────────────┼───────────────┼────┼──────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────┤
│         3 │  1756713176 │           1 │ users         │  2 │          │ Jane                                                                     │               │
├───────────┼─────────────┼─────────────┼───────────────┼────┼──────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────┤
│         4 │  1756713176 │           0 │ users         │  1 │  John  │         John Doe                                                                  │     John Doe │
├───────────┼─────────────┼─────────────┼───────────────┼────┼──────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────┤
│         5 │  1756713176 │          -1 │ users         │  2 │ Jane │                                                                              │               │
└───────────┴─────────────┴─────────────┴───────────────┴────┴──────────┴──────────────────────────────────────────────────────────────────────────────┴───────────────┘
turso>

```

If you modify your table schema (adding/dropping columns), the `table_columns_json_array()` function returns the current schema, not the historical one. This can lead to incorrect results when decoding older CDC records. Manually track schema versions by storing the output of `table_columns_json_array()` before making schema changes.
## Appendix A: Turso Internals

Turso's architecture resembles SQLite's but differs primarily in its
asynchronous I/O model. This asynchronous design enables applications to
leverage modern I/O interfaces like `io_uring,` maximizing storage device
performance. While an in-process database offers significant performance
advantages, integration with cloud services remains crucial for operations
like backups. Turso's asynchronous I/O model facilitates this by supporting
networked storage capabilities.

The high-level interface to Turso is the same as in SQLite:

* SQLite query language
* The `sqlite3_prepare()` function for translating SQL statements to programs
  ("prepared statements")
* The `sqlite3_step()` function for executing programs

If we start with the SQLite query language, you can use the `turso`
command, for example, to evaluate SQL statements in the shell:

```
turso> SELECT 'hello, world';
hello, world
```

To execute this SQL statement, the shell uses the `sqlite3_prepare()`
interface to parse the statement and generate a bytecode program, a step
called preparing a statement. When a statement is prepared, it can be executed
using the `sqlite3_step()` function.

To illustrate the different components of Turso, we can look at the sequence
diagram of a query from the CLI to the bytecode virtual machine (VDBE):

```mermaid
sequenceDiagram

participant main as cli/main
participant Database as core/lib/Database
participant Connection as core/lib/Connection
participant Parser as sql/mod/Parser
participant translate as translate/mod
participant Statement as core/lib/Statement
participant Program as vdbe/mod/Program

main->>Database: open_file
Database->>main: Connection
main->>Connection: query(sql)
Note left of Parser: Parser uses vendored sqlite3-parser
Connection->>Parser: next()
Note left of Parser: Passes the SQL query to Parser

Parser->>Connection: Cmd::Stmt (ast/mod.rs)

Note right of translate: Translates SQL statement into bytecode
Connection->>translate:translate(stmt)

translate->>Connection: Program 

Connection->>main: Ok(Some(Rows { Statement }))

note right of main: a Statement with <br />a reference to Program is returned

main->>Statement: step()
Statement->>Program: step()
Note left of Program: Program executes bytecode instructions<br />See https://www.sqlite.org/opcode.html
Program->>Statement: StepResult
Statement->>main: StepResult
```

To drill down into more specifics, we inspect the bytecode program for a SQL
statement using the `EXPLAIN` command in the shell. For our example SQL
statement, the bytecode looks as follows:

```
turso> EXPLAIN SELECT 'hello, world';
addr  opcode             p1    p2    p3    p4             p5  comment
----  -----------------  ----  ----  ----  -------------  --  -------
0     Init               0     4     0                    0   Start at 4
1     String8            0     1     0     hello, world   0   r[1]='hello, world'
2     ResultRow          1     1     0                    0   output=r[1]
3     Halt               0     0     0                    0
4     Transaction        0     0     0                    0
5     Goto               0     1     0                    0
```

The instruction set of the virtual machine consists of domain specific
instructions for a database system. Every instruction consists of an
opcode that describes the operation and up to 5 operands. In the example
above, execution starts at offset zero with the `Init` instruction. The
instruction sets up the program and branches to a instruction at address
specified in operand `p2`. In our example, address 4 has the
`Transaction` instruction, which begins a transaction. After that, the
`Goto` instruction then branches to address 1 where we load a string
constant `'hello, world'` to register `r[1]`. The `ResultRow` instruction
produces a SQL query result using contents of `r[1]`. Finally, the
program terminates with the `Halt` instruction.

### Frontend

#### Parser

The parser is the module in the front end that processes SQLite query language input data, transforming it into an abstract syntax tree (AST) for further processing. The parser is an in-tree fork of [lemon-rs](https://github.com/gwenn/lemon-rs), which in turn is a port of SQLite parser into Rust. The emitted AST is handed over to the code generation steps to turn the AST into virtual machine programs.

#### Code generator

The code generator module takes AST as input and produces virtual machine programs representing executable SQL statements. At high-level, code generation works as follows:

1. `JOIN` clauses are transformed into equivalent `WHERE` clauses, which simplifies code generation.
2. `WHERE` clauses are mapped into bytecode loops
3. `ORDER BY` causes the bytecode program to pass result rows to a sorter before returned to the application.
4. `GROUP BY` also causes the bytecode programs to pass result rows to an aggregation function before results are returned to the application.
  
#### Query optimizer

TODO

### Virtual Machine

TODO

### MVCC

The database implements a multi-version concurrency control (MVCC) using a hybrid architecture that combines an in-memory index with persistent storage through WAL (Write-Ahead Logging) and SQLite database files. The implementation draws from the Hekaton approach documented in Larson et al. (2011), with key modifications for durability handling.

The database maintains a centralized in-memory MVCC index that serves as the primary coordination point for all database connections. This index provides shared access across all active connections and stores the most recent versions of modified data. It implements version visibility rules for concurrent transactions following the Hekaton MVCC design. The architecture employs a three-tier storage hierarchy consisting of the MVCC index in memory as the primary read/write target for active transactions, a page cache in memory serving as an intermediate buffer for data retrieved from persistent storage, and persistent storage comprising WAL files and SQLite database files on disk.

_Read operations_ follow a lazy loading strategy with a specific precedence order. The database first queries the in-memory MVCC index to check if the requested row exists and is visible to the current transaction. If the row is not found in the MVCC index, the system performs a lazy read from the page cache. When necessary, the page cache retrieves data from both the WAL and the underlying SQLite database file.

_Write operations_ are handled entirely within the in-memory MVCC index during transaction execution. This design provides high-performance writes with minimal latency, immediate visibility of changes within the transaction scope, and isolation from other concurrent transactions until the transaction is committed.

_Commit operation_ ensures durability through a two-phase approach: first, the system writes the complete transaction write set from the MVCC index to the page cache, then the page cache contents are flushed to the WAL, ensuring durable storage of the committed transaction. This commit protocol guarantees that once a transaction commits successfully, all changes are persisted to durable storage and will survive system failures.

While the implementation follows Hekaton's core MVCC principles, it differs in one significant aspect regarding logical change tracking. Unlike Hekaton, this system does not maintain a record of logical changes after flushing data to the WAL. This design choice simplifies compatibility with the SQLite database file format.

### Pager

TODO

### I/O

Every I/O operation shall be tracked by a corresponding `Completion`. A `Completion` is just an object that tracks a particular I/O operation. The database `IO` will call it's complete callback to signal that the operation was complete, thus ensuring that every tracker can be poll to see if the operation succeeded.


To advance the Program State Machines, you must first wait for the tracked completions to complete. This can be done either by busy polling (`io.wait_for_completion`) or polling once and then yielding - e.g

  ```rust
  if !completion.is_completed {
    return StepResult::IO;
  }
  ```

This allows us to be flexible in places where we do not have the state machines in place to correctly return the Completion. Thus, we can block in certain places to avoid bigger refactorings, which opens up the opportunity for such refactorings in separate PRs.

To know if a function does any sort of I/O we just have to look at the function signature. If it returns `Completion`, `Vec<Completion>` or `IOResult`, then it does I/O.

The `IOResult` struct looks as follows:
  ```rust
  pub enum IOCompletions {
    Single(Arc<Completion>),
    Many(Vec<Arc<Completion>>),
  }

  #[must_use]
  pub enum IOResult<T> {
    Done(T),
    IO(IOCompletions),
  }
  ```

This implies that when a function returns an `IOResult`, it must be called again until it returns an `IOResult::Done` variant. This works similarly to how `Future`s are polled in rust. When you receive a `Poll::Ready(None)`, it means that the future stopped it's execution. In a similar vein, if we receive `IOResult::Done`, the function/state machine has reached the end of it's execution. `IOCompletions` is here to signal that, if we are executing any I/O operation, that we need to propagate the completions that are generated from it. This design forces us to handle the fact that a function is asynchronous in nature. This is essentially [function coloring](https://www.tedinski.com/2018/11/13/function-coloring.html), but done at the application level instead of the compiler level.

### Encryption

#### Goals

- Per-page encryption as an opt-in feature, so users don't have to compile/load the encryption extension
- All pages in db and WAL file to be encrypted on disk
- Least performance overhead as possible

#### Design

1. We do encryption at the page level, i.e., each page is encrypted and decrypted individually.
2. At db creation, we take key and cipher scheme information. We store the scheme information (also version) in the db file itself.
3. The key is not stored anywhere. So each connection should carry an encryption key. Trying to open a db with an invalid or empty key should return an error.
4. We generate a new randomized, cryptographically safe nonce every time we write a page.
5. We store the authentication tag and the nonce in the page itself.
6. We can support different cipher algorithms: AES, ChachaPoly, AEGIS, etc.
7. We can support key rotation. But rekeying would require writing the entire database.
8. We should also add import/export functionality to the CLI for better DX and compatibility with SQLite.

#### Metadata management

We store the nonce and tag (or the verification bits) in the page itself. During decryption, we will load these to decrypt and verify the data.

Example: Assume the page size is 4096 bytes and we use AEGIS 256. So we reserve the last 48 bytes
for the nonce (32 bytes) and tag (16 bytes).

```ignore
            Unencrypted Page              Encrypted Page
            ┌───────────────┐            ┌───────────────┐
            │               │            │               │
            │ Page Content  │            │   Encrypted   │
            │ (4048 bytes)  │  ────────► │    Content    │
            │               │            │ (4048 bytes)  │
            ├───────────────┤            ├───────────────┤
            │   Reserved    │            │    Tag (16)   │
            │  (48 bytes)   │            ├───────────────┤
            │   [empty]     │            │   Nonce (32)  │
            └───────────────┘            └───────────────┘
               4096 bytes                   4096 bytes
```

The above applies to all the pages except Page 1. Page 1 contains the SQLite header (the first 100 bytes). Specifically, bytes 16 to 24 contain metadata which is required to initialize the connection, which happens before we can set up the encryption context. So, we don't encrypt the header but instead use the header data as additional data (AD) for the encryption of the rest of the page. This provides us protection against tampering and corruption for the unencrypted portion.

On disk, the encrypted page 1 contains special bytes replacing SQLite's magic bytes (the
first 16 bytes):

```ignore
                   Turso Header (16 bytes)
       ┌─────────┬───────┬────────┬──────────────────┐
       │         │       │        │                  │
       │ "Turso" │Version│ Cipher │     Unused       │
       │  (5)    │ (1)   │  (1)   │    (9 bytes)     │
       │         │       │        │                  │
       └─────────┴───────┴────────┴──────────────────┘
        0-4      5       6        7-15

       Standard SQLite Header: "SQLite format 3\0" (16 bytes)
                           ↓
       Turso Encrypted Header: "Turso" + Version + Cipher ID + Unused
```

The current version is 0x00. The cipher IDs are:

| Algorithm | Cipher ID |
|-----------|-----------|
| AES-GCM (128-bit) | 1 |
| AES-GCM (256-bit) | 2 |
| AEGIS-256 | 3 |
| AEGIS-256-X2 | 4 |
| AEGIS-256-X4 | 5 |
| AEGIS-128L | 6 |
| AEGIS-128-X2 | 7 |
| AEGIS-128-X4 | 8 |

#### Future work
1. I have omitted the key derivation aspect. We can later add passphrase and key derivation support.
2. Pages in memory are unencrypted. We can explore memory enclaves and other mechanisms.

#### Other Considerations

You may check the [RFC discussion](https://github.com/tursodatabase/turso/issues/2447) and also [Checksum RFC discussion](https://github.com/tursodatabase/turso/issues/2178) for the design decisions.

- SQLite has some unused bytes in the header left for future expansion. We considered using this portion to store the cipher information metadata but decided not to because these may get used in the future.
- Another alternative was to truncate tag bytes of page 1, then store the meta information. Ultimately, it seemed much better to store the metadata in the magic bytes.
- For per-page metadata, we decided to store it in the reserved space. The reserved space is for extensions; however, I could not find any usage of it other than the official Checksum shim and other encryption extensions.

# tomar en cuenta a restricciones y obligatorias,  antes de implementar cualquier funcionalidad (modulos, funcionalidades complejas) debes  examinar el proyecto y asegurarte que lo que vayas a crear no este ya implementado y si esta debes actualizarlo a los requerimientos, actuales 

# -Todo el desarrollo lo debes hacer respetando las buenas practicas de desarrollo de software bajo django  y python 

# - por cada 3 cambios debes obligatoriamente volver al contexto para que no pierdas el enfoque de lo que estas creando y tenga logica, completar los flujos y la logica para completar las funciones existentes, si no lo haces el proyecto se volvera un desastre 