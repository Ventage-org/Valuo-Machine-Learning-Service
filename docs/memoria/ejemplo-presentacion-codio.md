4.3	Implementación Aplicación / Solución (Código)
Backend:
El backend de FocusGuard fue desarrollado utilizando FastAPI y organizado bajo una arquitectura modular basada en dominios funcionales y principios de separación de responsabilidades. Cada módulo encapsula una funcionalidad específica del sistema y se estructura en capas independientes para la gestión de controladores, servicios de negocio, acceso a datos, modelos y esquemas de validación. Esta organización facilita la mantenibilidad, escalabilidad y evolución del sistema, permitiendo incorporar nuevas funcionalidades sin afectar significativamente los componentes existentes.
La estructura general de módulos implementada en el backend se presenta a continuación:
app/
└── modules/
├── users/
├── goals/
├── objectives/
├── websites/
├── sites/
├── contents/
├── categories/
├── configuration/
├── reports/
├── ml/
├── ml_classification/
└── interventions/
Cada módulo sigue una estructura interna común basada en capas, como se muestra en el siguiente ejemplo:
goals/
├── controllers/
│ └── goal_controller.py
├── services/
│ └── goal_service.py
├── implementation/
│ └── goal_repository.py
├── models/
│ └── goal_model.py
└── schemas/
└── goal_schema.py
Donde la carpeta controllers expone los endpoints REST del sistema, services implementa la lógica de negocio, implementation gestiona el acceso a datos mediante repositorios, models define las entidades persistentes y schemas contiene los modelos de validación y transferencia de datos utilizados por la API.

 Controlador de Categorías de Sitios Web (category_website_controller.py)
router = APIRouter(prefix="/categories/web", tags=["categories-web"])


def get_service(db: AsyncSession = Depends(get_db)) -> CategoryWebsiteService:
    repo = CategoryWebsiteRepository(db_session=db)
    return CategoryWebsiteService(repo=repo)


@router.post("/", response_model=CategoryWebsiteResponse, status_code=status.HTTP_201_CREATED)
async def create_category_website(
    data: CategoryWebsiteCreate,
    service: CategoryWebsiteService = Depends(get_service),
):
    try:
        return await service.create_category_website(data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

Este componente se encarga de la gestión y clasificación de las plataformas web que el sistema utiliza para evaluar el comportamiento del usuario. Su función es administrar el catálogo de categorías, permitiendo la creación de nuevas clasificaciones y la eliminación de aquellas que ya no sean requeridas. Asimismo, facilita la búsqueda y filtrado de estas categorías mediante criterios clave como su identificador, su nombre, su código interno o el nivel de peso asignado, el cual determina el impacto o relevancia de cada sitio web dentro del análisis de productividad del proyecto.

 Controlador de Categorías de Contenido (category_content_controller.py)
router = APIRouter(prefix="/categories/content", tags=["categories-content"])


def get_service(db: AsyncSession = Depends(get_db)) -> CategoryContentService:
    repo = CategoryContentRepository(db_session=db)
    return CategoryContentService(repo=repo)


@router.post("/", response_model=CategoryContentResponse, status_code=status.HTTP_201_CREATED)
async def create_category_content(
    data: CategoryContentCreate,
    service: CategoryContentService = Depends(get_service),
):
    try:
        return await service.create_category_content(data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

Este componente es el encargado de gestionar la clasificación del contenido específico dentro de los sitios web o aplicaciones que consume el usuario, permitiendo un análisis más granular que el de la simple URL. Su función consiste en administrar el catálogo de estas etiquetas de contenido, controlando la creación de nuevas clasificaciones y la eliminación de aquellas obsoletas. Asimismo, provee los mecanismos para consultar y filtrar dichas categorías a partir de su identificador único, nombre, código referencial o el peso asignado, el cual sirve para ponderar el nivel de distracción o productividad de los textos o temáticas detectadas durante el monitoreo.

 Controlador de Tiempos de Descanso (rest_time_controller.py)
router = APIRouter(prefix="/rest-time", tags=["rest-time"])

@router.get("/{user_id}", response_model=RestTimeResponse)
async def get_tiempo_descanso(
    user_id: int,
    db: AsyncSession = Depends(get_db)
):
    try:
        service = RestTimeService(db)
        return await service.get_tiempo_descanso(user_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo tiempo de descanso: {str(e)}"
        )


@router.put("/{user_id}", response_model=RestTimeResponse)
async def update_tiempo_descanso(
    user_id: int,
    tiempo_update: RestTimeUpdate,
    db: AsyncSession = Depends(get_db)
):
    try:
        service = RestTimeService(db)
        return await service.update_tiempo_descanso(user_id, tiempo_update)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error actualizando tiempo de descanso: {str(e)}"
        )

Este componente se encarga de administrar los parámetros de tiempo destinados a las pausas y periodos de recuperación que el usuario debe tomar durante sus jornadas de actividad. Su función principal es garantizar la personalización de estos intervalos, permitiendo consultar la configuración actual del tiempo de descanso asignado a un usuario específico, así como modificar o actualizar dichos valores según sus necesidades individuales o estrategias de autorregulación. Esto contribuye directamente a equilibrar las sesiones de trabajo continuo y mitigar la fatiga cognitiva mediante pausas controladas por el sistema.

 Controlador de Contenidos (content_controller)
router = APIRouter(prefix="/contents", tags=["contents"])

def get_service(db: AsyncSession = Depends(get_db)) -> ContentService:
    repo = ContentRepository(db_session=db)
    return ContentService(repo=repo)

@router.post("/", response_model=ContentResponse, status_code=status.HTTP_201_CREATED)
async def create_content(
    data: ContentCreate,
    service: ContentService = Depends(get_service),
) -> ContentResponse:
    try:
        return await service.create_content(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.get("/", response_model=list[ContentResponse])
async def get_all_contents(
    service: ContentService = Depends(get_service),
) -> list[ContentResponse]:
    return await service.get_contents()

Este componente se encarga de gestionar los registros individuales de contenido específico que son detectados o consumidos en las plataformas digitales durante el monitoreo. Su función consiste en administrar el ciclo de vida de estos elementos de información, permitiendo el almacenamiento de nuevos contenidos identificados y la eliminación de aquellos que ya no aporten valor al historial. Asimismo, facilita la consulta global o el filtrado por medio de un identificador único, sirviendo como base operativa para que el sistema reconozca qué temáticas exactas están presentes en la actividad del usuario y evaluar su pertinencia en el entorno de trabajo.

 Controlador de Interacciones de Contenido por Usuario (content_user_controller.py)
router = APIRouter(prefix="/content-users", tags=["content-users"])
def get_service(db: AsyncSession = Depends(get_db)) -> ContentUserService:
    repo = ContentUserRepository(db_session=db)
    user_repo = UserRepository(db_session=db)
    content_repo = ContentRepository(db_session=db)
    category_content_repo = CategoryContentRepository(db_session=db)
    website_user_repo = WebsiteUserRepository(db_session=db)
    return ContentUserService(
        repo=repo,
        user_repo=user_repo,
        content_repo=content_repo,
        category_content_repo=category_content_repo,
        website_user_repo=website_user_repo
    )
@router.post("/", response_model=ContentUserResponse, status_code=status.HTTP_201_CREATED)
async def create_content_user(
    data: ContentUserCreate,
    service: ContentUserService = Depends(get_service),
) -> ContentUserResponse:
    try:
        return await service.create(data)
    except NotFoundException as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

Este componente tiene como propósito vincular y contextualizar el contenido específico que consume un usuario con las plataformas y clasificaciones correspondientes dentro del sistema. Su función principal es registrar, actualizar y auditar las interacciones individuales de cada usuario, asociando directamente el texto o información extraída con el sitio web de origen y su categoría temática. Asimismo, provee herramientas avanzadas de consulta que permiten segmentar este historial de navegación cruzando múltiples variables clave, tales como las actividades de un usuario en un sitio web específico, los contenidos clasificados bajo una misma categoría, o el comportamiento general de un estudiante, consolidando así el insumo fundamental para la detección detallada de patrones de comportamiento.

 Controlador de Contenidos Visitados (content_visited_controller.py)
router = APIRouter(prefix="/content-visited", tags=["content_visited"])

def get_service(db: AsyncSession = Depends(get_db)) -> ContentVisitedService:
    repo = ContentVisitedRepository(db_session=db)
    user_repo = UserRepository(db_session=db)
    content_user_repo = ContentUserRepository(db_session=db)
    return ContentVisitedService(
        repo=repo,
        user_repo=user_repo,
        content_user_repo=content_user_repo,
    )

@router.post("/", response_model=ContentVisitedResponse, status_code=status.HTTP_201_CREATED)
async def create_content_visited(
    data: ContentVisitedCreate,
    service: ContentVisitedService = Depends(get_service),
) -> ContentVisitedResponse:
    try:
        return await service.create(data)
    except NotFoundException as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

Este componente se encarga de registrar y auditar el tiempo de permanencia e interacción del usuario con un contenido específico en tiempo real. Su función principal es llevar la trazabilidad cronológica de las sesiones de navegación, almacenando el momento exacto en el que un usuario accede a una sección informativa y actualizando de forma parcial dicho registro cuando se detecta su salida o el cierre de la pestaña. Asimismo, provee los mecanismos de consulta para evaluar el historial completo de visitas o segmentarlo en base a un usuario en particular y sus interacciones asociadas, permitiendo cuantificar con precisión la duración exacta de la exposición a contenidos potencialmente distractores o productivos.

 Controlador de Metas (goal_controller.py)
router = APIRouter(prefix="/goals", tags=["goals"])

async def get_service(db: AsyncSession = Depends(get_db)) -> GoalService:
    repo = GoalRepository(db_session=db)
    return GoalService(repo=repo)

@router.get("/", response_model=List[GoalResponse])
async def get_all_goals(service: GoalService = Depends(get_service)):

    return await service.get_all()

@router.get("/{goal_id}", response_model=GoalResponse)
async def get_goal_by_id(goal_id: int, service: GoalService = Depends(get_service)):

    try:
        return await service.get_by_id(goal_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

Este componente se encarga de administrar los objetivos de autorregulación y productividad establecidos para los usuarios dentro del sistema. Su función principal es gestionar el ciclo de vida de estas metas, permitiendo la creación de nuevos compromisos de cumplimiento y la modificación o eliminación de objetivos existentes conforme evolucionen las necesidades del usuario. Asimismo, facilita el seguimiento del rendimiento individual al proveer mecanismos de consulta para recuperar el listado completo de metas registradas o filtrar de manera específica aquellas pertenecientes a un estudiante en particular, sirviendo como el marco de referencia contra el cual se contrastan los hábitos de navegación detectados.

 Controlador de Progreso Diario (daily_progress_controller.py)
router = APIRouter(prefix="/daily-progress", tags=["daily_progress"])

async def get_service(db: AsyncSession = Depends(get_db)) -> DailyProgressService:
    repo = DailyProgressRepository(db_session=db)
    return DailyProgressService(repo=repo)

async def get_weekly_goal_service(db: AsyncSession = Depends(get_db)) -> WeeklyGoalService:
    repo = WeeklyGoalRepository(db_session=db)
    return WeeklyGoalService(repo=repo)

@router.get("/", response_model=List[DailyProgressResponse])
async def get_all_daily_progress(service: DailyProgressService = Depends(get_service)):

    return await service.get_all()

@router.get("/{progress_id}", response_model=DailyProgressResponse)
async def get_daily_progress_by_id(progress_id: int, service: DailyProgressService = Depends(get_service)):

    try:
        return await service.get_by_id(progress_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

Este componente se encarga de monitorear, cuantificar y evaluar el avance cotidiano de los usuarios respecto a sus objetivos de rendimiento programados. Su función principal es auditar las métricas de tiempo invertido por día en actividades productivas o de estudio, procesando de manera automática si dicho esfuerzo ha sido suficiente para alcanzar las metas semanales vinculadas. Asimismo, facilita la consolidación de reportes de desempeño al proveer mecanismos para consultar la evolución del usuario, obtener el estado de la meta del día en curso y generar resúmenes periódicos que permiten identificar si el estudiante mantiene una constancia regular o si requiere de alertas e intervenciones debido a desviaciones en sus planes.

 Lógica de Evaluación y Métricas de Cumplimiento de Metas (daily_progress_service.py)
async def register_daily_progress(self, goal_id: int, tiempo_usado: int, weekly_goal) -> DailyProgressResponse:
        if weekly_goal.opcion_1 == 1:
            es_alcanzado = tiempo_usado >= weekly_goal.tiempo
        else:  
            es_alcanzado = tiempo_usado <= weekly_goal.tiempo
        
        today = datetime.now()
        existing_progress = await self.repo.get_today_progress(goal_id, today)
        
        if existing_progress:
            return await self.update(existing_progress.id, DailyProgressUpdate(
                tiempo_alcanzado=tiempo_usado, es_alcanzado=es_alcanzado
            ))
        else:
            return await self.create(DailyProgressCreate(
                id_objetivos_semanales=goal_id, tiempo_alcanzado=tiempo_usado, es_alcanzado=es_alcanzado
            ))

Este fragmento de la capa de servicios implementa las reglas de negocio encargadas de auditar y evaluar cuantitativamente el progreso del usuario frente a sus metas académicas establecidos. El método register_daily_progress evalúa condicionalmente el estado del objetivo mediante un parámetro lógico (opcion_1), permitiendo discernir de manera flexible si la meta consiste en maximizar el tiempo invertido en plataformas productivas (>=) o en restringir y disminuir el tiempo dentro de sitios distractivos (<=). Asimismo, automatiza la persistencia evitando la duplicidad de registros mediante una estrategia de inserción o actualización (upsert) basada en la fecha del sistema.

 Controlador de Objetivos Semanales (weekly_goal_controller.py)
router = APIRouter(prefix="/weekly-goals", tags=["weekly_goals"])

async def get_service(db: AsyncSession = Depends(get_db)) -> WeeklyGoalService:
    repo = WeeklyGoalRepository(db_session=db)
    return WeeklyGoalService(repo=repo)

@router.get("/", response_model=List[WeeklyGoalResponse])
async def get_all_weekly_goals(service: WeeklyGoalService = Depends(get_service)):

    return await service.get_all()

@router.get("/{goal_id}", response_model=WeeklyGoalResponse)
async def get_weekly_goal_by_id(goal_id: int, service: WeeklyGoalService = Depends(get_service)):

    try:
        return await service.get_by_id(goal_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

Este componente se encarga de estructurar y auditar la planificación a mediano plazo de los usuarios mediante la definición de metas con una vigencia e impacto semanal. Su función principal es administrar la creación, actualización y eliminación de estos objetivos de rendimiento, permitiendo además que el sistema los marque formalmente como completados al concluir satisfactoriamente el ciclo estipulado. Asimismo, centraliza el análisis de cumplimiento del estudiante al proveer herramientas de consulta que calculan el porcentaje exacto de progreso acumulado, ya sea para un objetivo individual o para el conjunto global de metas asociadas a un usuario, consolidando una métrica indispensable para medir la efectividad en el control de la procrastinación.

 Controlador de Reportes de Horas de Ocio Semanal (leisure_time_controller.py)
router = APIRouter(
    prefix="/reports/weekly-leisure-hours",
    tags=["leisure_time"]
)

async def get_service(db: AsyncSession = Depends(get_db)) -> LeisureTimeService:
    repo = LeisureTimeRepository(db_session=db)
    return LeisureTimeService(repo=repo)

@router.get("/", response_model=List[LeisureTimeResponse])
async def get_leisure_time(
    user_id: int,
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    service: LeisureTimeService = Depends(get_service)
):

    try:
        return await service.get_leisure_time(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

Este componente se encarga de consolidar y estructurar la información analítica sobre los periodos de inactividad o distracción del usuario para la generación de reportes periódicos. Su función principal es calcular y agrupar el tiempo de ocio acumulado por cada día, permitiendo acotar este análisis dentro de un rango de fechas específico solicitado por el usuario o establecido de manera semanal. De este modo, el sistema provee una métrica cuantitativa clara sobre los intervalos dedicados a plataformas no productivas, sirviendo como un insumo de diagnóstico fundamental para que el estudiante evalúe de manera objetiva sus patrones de procrastinación a lo largo del tiempo.

 Controlador de Reportes de Sitios Más Visitados (top_site_controller.py)
router = APIRouter(prefix="/reports/top-sites", tags=["top_sites"])

async def get_service(db: AsyncSession = Depends(get_db)) -> TopSiteService:
    repo = TopSiteRepository(db_session=db)
    return TopSiteService(repo=repo)

@router.get("/", response_model=List[TopSiteResponse])
async def get_top_sites(
    user_id: int,
    service: TopSiteService = Depends(get_service)
):
    try:
        return await service.get_top_sites(user_id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

Este componente se encarga de procesar e identificar las principales fuentes de distracción o consumo de tiempo por parte del usuario para la generación de reportes analíticos. Su función principal es calcular y consolidar el listado de las cinco plataformas web en las que el estudiante registra mayor permanencia, ordenándolas de manera descendente en función del tiempo total acumulado. Al aislar y jerarquizar estos sitios de mayor recurrencia, el sistema provee un indicador crítico de "procrastinación real", permitiendo visibilizar con precisión cuáles son los entornos digitales que impactan con mayor severidad en el desvío de sus objetivos académicos.

 Controlador de Usuarios y Autenticación (user_controller.py)
router = APIRouter(prefix="/users", tags=["users"])

def get_service(db: AsyncSession = Depends(get_db)) -> UserService:
    return UserService(db)

@router.get("/me")
async def me(user_id: str = Depends(get_current_user)):
    return {"user_id": user_id}

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> UserResponse:
    service = UserService(db)
    try:
        return await service.create_user(user_data)
    except BusinessException as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DatabaseException as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

Este componente se encarga de centralizar la administración de las cuentas de usuario y de salvaguardar el acceso al sistema. Su función principal es gestionar el ciclo de vida de las credenciales de los estudiantes, abarcando desde el registro de nuevas cuentas, la actualización de perfiles y la eliminación de registros, hasta la búsqueda de perfiles específicos mediante identificadores o correos electrónicos. Asimismo, es el responsable de aplicar las reglas de seguridad esenciales del proyecto, controlando la validación del inicio de sesión con tokens de acceso seguro, la identificación del usuario activo en tiempo real y el flujo de recuperación o restablecimiento de contraseñas en caso de olvido, garantizando que toda la información de productividad recopilada permanezca protegida.


 Controlador de Sitios Web Visitados (website_visited_controller.py)
router = APIRouter(prefix="/website-visited", tags=["website_visited"])

def get_service(db: AsyncSession = Depends(get_db)) -> WebsiteVisitedService:
    repo = WebsiteVisitedRepository(db_session=db)
    user_repo = UserRepository(db_session=db)
    website_user_repo = WebsiteUserRepository(db_session=db)
    return WebsiteVisitedService(
        repo=repo,
        user_repo=user_repo,
        website_user_repo=website_user_repo,
    )

@router.post("/", response_model=WebsiteVisitedResponse, status_code=status.HTTP_201_CREATED)
async def create_visit(
    data: WebsiteVisitedCreate,
    service: WebsiteVisitedService = Depends(get_service),
) -> WebsiteVisitedResponse:
    try:
        return await service.create(data)
    except NotFoundException as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

Este archivo define el controlador de la API (enrutador de FastAPI) encargado de gestionar el historial de navegación del usuario. Utiliza inyección de dependencias mediante la función get_service para inicializar el servicio de negocio (WebsiteVisitedService) junto con los repositorios asíncronos de la base de datos de usuarios y sitios web.
El código expone endpoints clave para el funcionamiento de la extensión: create_visit (POST), que registra el momento exacto en el que un usuario ingresa a un sitio web, y update_exit_time (PATCH), encargado de actualizar la hora de salida de la página visitada.

 Lógica de Negocio y Validación de Visitas Web (website_visited_service.py)
class WebsiteVisitedService:
    def __init__(
            self,
            repo: WebsiteVisitedRepository,
            user_repo: UserRepository,
            website_user_repo: WebsiteUserRepository
    ) -> None:
        self.repo = repo
        self.user_repo = user_repo
        self.website_user_repo = website_user_repo

    async def create(self, data: WebsiteVisitedCreate) -> WebsiteVisitedResponse:
        user = await self.user_repo.get_by_id(data.id_usuarios)
        if user is None:
            raise NotFoundException("El ID de usuario proporcionado no existe.")

        website_user = await self.website_user_repo.get_by_id(data.id_sitios_web_usuario)
        if website_user is None:
            raise NotFoundException("El ID de sitiosweb_usuario proporcionado no existe.")

        registro = await self.repo.create(data)
        return WebsiteVisitedResponse.model_validate(registro)

Este servicio centraliza las reglas de negocio para el control del tiempo de navegación. En el método create, valida que tanto el usuario como el sitio web asociado existan realmente en la base de datos antes de guardar la visita, actuando como una capa de integridad de datos. En el método update_exit_time, el servicio maneja la conversión de zonas horarias (timezone-aware) para comparar de manera segura la marca de tiempo de entrada con la de salida enviada por la extensión web. Esto previene incongruencias en las métricas del sistema al asegurar mediante una excepción ValueError que un usuario no pueda registrar un tiempo de salida anterior al de su ingreso.

 Persistencia Relacional y Filtros Temporales de Intervenciones (intervention_repository.py)
async def get_today_count_by_user(self, user_id: int) -> int:
        stmt = select(func.count(InterventionModel.id)).where(
            InterventionModel.id_usuarios == user_id,
            func.date(InterventionModel.fecha_despliegue) == datetime.now().date(),
        )
        return int((await self.db.scalar(stmt)) or 0)

    async def get_last_by_user_if_today(self, user_id: int) -> Optional[InterventionModel]:
        stmt = (
            select(InterventionModel)
            .where(InterventionModel.id_usuarios == user_id)
            .order_by(InterventionModel.fecha_despliegue.desc())
            .limit(1)
        )
        registro = (await self.db.execute(stmt)).scalar_one_or_none()

        if registro is None:
            return None

        if registro.fecha_despliegue.date() != datetime.now().date():
            return None

        return registro

Este fragmento de la capa de acceso a datos (repositorio) implementa la lógica de persistencia asíncrona mediante SQLAlchemy. Destaca por el uso de funciones integradas del ORM como func.count y func.date para realizar el cálculo del comportamiento del usuario directamente en el servidor de datos, optimizando el rendimiento de la aplicación.
El método get_today_count_by_user ejecuta una consulta agregada que cuantifica las intervenciones desplegadas emparejando la fecha del sistema (datetime.now().date()), permitiendo evaluar la reincidencia en la sesión actual. Por otra parte, get_last_by_user_if_today combina ordenamiento descendente (order_by) y truncamiento de resultados (limit(1)) para validar de forma eficiente si la última acción correctiva registrada pertenece al día en curso, garantizando la continuidad de las políticas anti-procrastinación.

 Extracción de Características Temporales para transferir al modelo secuencial (prediction_repository.py)
async def get_features_by_user(self, user_id: int):
        query = text("""
        SELECT
            CASE 
                WHEN c.nombre = 'Distractivo' THEN 'Distractor'
                ELSE c.nombre
            END AS categoria,
            GREATEST(TIMESTAMPDIFF(SECOND, v.fecha_hora_ingreso, COALESCE(v.fecha_hora_salida, NOW())), 1) AS duracion_segundos
        FROM sitios_web_visitados v
        JOIN sitios_web_usuario u ON v.id_sitios_web_usuario = u.id
        JOIN categorias_web c ON u.id_categorias_web = c.id
        WHERE v.id_usuarios = :user_id AND v.fecha_hora_salida IS NOT NULL
        ORDER BY v.fecha_hora_ingreso DESC LIMIT 10
        """)

        result = await self.session.execute(query, {"user_id": user_id})
        rows = result.fetchall()

        return [{"categoria": r.categoria, "duracion_segundos": float(r.duracion_segundos)} for r in rows]

Este método asíncrono constituye el núcleo analítico del repositorio, encargado de estructurar el vector de características (features) que alimenta al modelo predictivo. Mediante una consulta SQL nativa optimizada, se extraen los últimos diez sitios web visitados por el usuario para reconstruir su secuencia de comportamiento reciente.
La lógica implementa la función TIMESTAMPDIFF para calcular el tiempo neto de navegación en segundos y mitiga posibles inconsistencias temporales mediante COALESCE (asumiendo la hora actual si el usuario sigue en el sitio) y GREATEST, asegurando un umbral mínimo de un segundo. El resultado consolida un conjunto de datos estandarizado y ordenado cronológicamente que define de forma cuantitativa si el comportamiento previo del estudiante se orienta a la productividad o a la procrastinación.
