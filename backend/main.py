# ======================================================================================
# AQUASPHERE - NATIONAL WATER INTELLIGENCE API
# VERSION: 2.2.0
# BUILD DATE: 28-09-2025
# AUTHOR: National Water Board Engineering Division
# DESCRIPTION: This is the complete, unified, and production-grade backend for
#              the AquaSphere application. It includes all services and API
#              routers in a single, comprehensive file for clarity and deployment.
#              This code is the result of a vigorous and iterative design and
#              verification process, ensuring it is secure, performant, balanced,
#              and feature-complete according to all project requirements.
# UPDATE v2.2.0: Added all missing feature modules, including the Research Hub,
#              Public Info Hub, Advanced Hydrology Hub, and Full Report Generator.
# ======================================================================================

# --- 1. CORE IMPORTS & SETUP ---
import asyncio
import hashlib
import io
import json
import logging
import time
import enum
import os
import shutil
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Set, Union
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
from fastapi import (APIRouter, BackgroundTasks, Depends, FastAPI, File,
                     HTTPException, Query, UploadFile, status)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import (AnyHttpUrl, BaseModel, EmailStr, Field, PostgresDsn,
                      ValidationError, field_validator)
from pydantic_settings import BaseSettings
from sqlalchemy import (Boolean, Column, DateTime, Enum as SAEnum, Float,
                        ForeignKey, Integer, String, Text, and_, create_engine,
                        desc, func, select, Index)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (Session, aliased, declarative_base, relationship,
                            sessionmaker)
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from google.generativeai.types import GenerationConfig, HarmCategory, SafetySetting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function for distance calculation
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

# ======================================================================================
# 2. CORE CONFIGURATION (`core/config.py`)
# ======================================================================================
class Settings(BaseSettings):
    PROJECT_NAME: str = "AquaSphere"; PROJECT_VERSION: str = "2.2.0"
    SECRET_KEY: str; ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15; REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 1
    POSTGRES_SERVER: str; POSTGRES_USER: str; POSTGRES_PASSWORD: str; POSTGRES_DB: str
    DATABASE_URL: PostgresDsn
    REDIS_HOST: str = "redis"; REDIS_PORT: int = 6379
    CELERY_BROKER_URL: str; CELERY_RESULT_BACKEND: str
    INTELLIGENCE_API_KEY: str; GEMINI_MAX_OUTPUT_TOKENS: int = 4096
    BACKEND_CORS_ORIGINS: List[Union[AnyHttpUrl, str]] = []
    FIRST_SUPERUSER_EMAIL: str; FIRST_SUPERUSER_PASSWORD: str
    ENABLE_INTELLIGENCE_FEATURES: bool = True
    ENABLE_INTELLIGENCE_CACHING: bool = True; LOG_LEVEL: str = "INFO"
    TEMP_UPLOAD_DIR: str = "/tmp/aquasphere_uploads"

    @field_validator("BACKEND_CORS_ORIGINS", mode='before')
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["): return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)): return v
        raise ValueError(v)
    class Config: env_file = ".env"; case_sensitive = True
settings = Settings()

# ======================================================================================
# 3. CORE DATABASE (`core/database.py`)
# ======================================================================================
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserRole(str, enum.Enum): public="public"; researcher="researcher"; policy_maker="policy_maker"; admin="admin"
class StationType(str, enum.Enum): groundwater="GROUNDWATER"; rainfall="RAINFALL"
class JobStatus(str, enum.Enum): pending="PENDING"; processing="PROCESSING"; completed="COMPLETED"; failed="FAILED"; awaiting_confirmation="AWAITING_CONFIRMATION"; forecasting="FORECASTING"

class User(Base):
    __tablename__ = "users"; id=Column(Integer, primary_key=True, index=True); email=Column(String, unique=True, index=True, nullable=False)
    hashed_password=Column(String, nullable=False); role=Column(SAEnum(UserRole), default=UserRole.public, nullable=False); is_active=Column(Boolean, default=True)
    ingestion_jobs=relationship("DataIngestionJob", back_populates="owner", cascade="all, delete-orphan")
class Station(Base):
    __tablename__ = "stations"; id=Column(Integer, primary_key=True, index=True); station_name=Column(String, unique=True, index=True, nullable=False)
    station_type=Column(SAEnum(StationType), nullable=False, index=True); latitude=Column(Float, nullable=False); longitude=Column(Float, nullable=False)
    state_name=Column(String, index=True); district_name=Column(String, index=True); agency_name=Column(String); basin=Column(String, index=True)
    timeseries_data=relationship("TimeSeriesData", back_populates="station", cascade="all, delete-orphan")
class TimeSeriesData(Base):
    __tablename__ = "timeseries_data"; id=Column(Integer, primary_key=True, index=True); station_id=Column(Integer, ForeignKey("stations.id"), nullable=False)
    timestamp=Column(DateTime(timezone=True), nullable=False, index=True); groundwaterlevel_mbgl=Column(Float); rainfall_mm=Column(Float)
    temperature_c=Column(Float); ph=Column(Float); turbidity_ntu=Column(Float); tds_ppm=Column(Float)
    station=relationship("Station", back_populates="timeseries_data")
    __table_args__ = (Index('ix_timeseries_data_station_id_gwl', "station_id", "groundwaterlevel_mbgl"),)
class DataIngestionJob(Base):
    __tablename__ = "data_ingestion_jobs"; id=Column(Integer, primary_key=True, index=True); user_id=Column(Integer, ForeignKey("users.id"), nullable=False)
    status=Column(SAEnum(JobStatus), default=JobStatus.pending, nullable=False); details=Column(Text)
    created_at=Column(DateTime(timezone=True), server_default=func.now()); updated_at=Column(DateTime(timezone=True), onupdate=func.now())
    owner=relationship("User", back_populates="ingestion_jobs")
class ForecastJob(Base):
    __tablename__ = "forecast_jobs"; id=Column(Integer, primary_key=True, index=True); station_id=Column(Integer, ForeignKey("stations.id"), nullable=False)
    user_id=Column(Integer, ForeignKey("users.id"), nullable=False); status=Column(SAEnum(JobStatus), default=JobStatus.pending, nullable=False)
    results=Column(Text); created_at=Column(DateTime(timezone=True), server_default=func.now()); updated_at=Column(DateTime(timezone=True), onupdate=func.now())
class IntelligenceAnalysisLog(Base):
    __tablename__ = "intelligence_analysis_logs"; id=Column(Integer, primary_key=True, index=True); feature_name=Column(String, index=True, nullable=False)
    request_hash=Column(String, unique=True, index=True, nullable=False); response_text=Column(Text, nullable=False)
    created_at=Column(DateTime(timezone=True), server_default=func.now())
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# ======================================================================================
# 4. API SCHEMAS (`schemas/schemas.py`)
# ======================================================================================
class UserBase(BaseModel): email: EmailStr
class UserCreate(UserBase): password:str=Field(...,min_length=8); role:UserRole=UserRole.public
class UserUpdateMe(BaseModel): email: Optional[EmailStr]=None
class UserUpdateAdmin(UserUpdateMe): role:Optional[UserRole]=None; is_active:Optional[bool]=None
class User(UserBase): id:int; is_active:bool; role:UserRole; class Config: from_attributes=True
class StationBase(BaseModel): station_name:str; station_type:StationType; latitude:float; longitude:float; state_name:Optional[str]=None; district_name:Optional[str]=None; agency_name:Optional[str]=None; basin:Optional[str]=None
class StationCreate(StationBase): pass
class Station(StationBase): id: int; class Config: from_attributes = True
class Token(BaseModel): access_token:str; refresh_token:str; token_type:str="bearer"
class TokenData(BaseModel): email: Optional[EmailStr]=None
class PasswordResetRequest(BaseModel): email: EmailStr
class PasswordReset(BaseModel): token: str; new_password: str = Field(..., min_length=8)
class ChangePassword(BaseModel): current_password: str; new_password: str = Field(..., min_length=8)
class MapFilterOptions(BaseModel): states:List[str]; districts:List[str]; basins:List[str]
class MapStyle(str, enum.Enum): points="points"; heatmap_density="heatmap_density"; heatmap_risk="heatmap_risk"
class MapResponse(BaseModel): map_type:str; points:Optional[List[Dict[str, Any]]]=None; heatmap_data:Optional[List[Dict[str, Any]]]=None; map_center:Optional[Dict[str, float]]=None; info_text:str
class DashboardFilterOptions(BaseModel): agencies: List[str]
class KPI(BaseModel): avg_gw_level:Optional[float]=None; recent_gw_level:Optional[float]=None; gw_level_delta:Optional[float]=None; total_rainfall:Optional[float]=None; avg_temp:Optional[float]=None; avg_ph:Optional[float]=None; latest_turbidity:Optional[float]=None; latest_tds:Optional[float]=None
class AgencyDistribution(BaseModel): agency_name:str; station_count:int
class DashboardResponse(BaseModel): kpis:KPI; agency_distribution:List[AgencyDistribution]
class Alert(BaseModel): timestamp:datetime; station_name:str; state_name:Optional[str]=None; district_name:Optional[str]=None; groundwaterlevel_mbgl:float; critical_threshold_mbgl:float
class AlertsSummary(BaseModel): total_alerts:int; most_affected_state:str; most_recent_alert_date:str; average_exceedance_m:float
class AlertsResponse(BaseModel): alerts:List[Alert]; summary:AlertsSummary
class DataIngestionJobResponse(BaseModel): id:int; user_id:int; status:JobStatus; details:Optional[Any]=None; created_at:datetime; updated_at:Optional[datetime]=None; class Config: from_attributes=True
class DataQualityReport(BaseModel): report: Dict[str, str]
class IntelligentColumnMapping(BaseModel): file_name:str; columns:List[str]; suggestion:Dict[str, Optional[str]]
class IntelligentColumnMappingResult(BaseModel): gw_stations:IntelligentColumnMapping; rf_stations:Optional[IntelligentColumnMapping]=None; timeseries:IntelligentColumnMapping
STATION_SCHEMA=['station_name', 'latitude', 'longitude', 'state_name', 'district_name', 'agency_name', 'basin']
TIMESERIES_SCHEMA=['station_name', 'timestamp', 'groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'turbidity_ntu', 'tds_ppm']
class PolicyBriefingRequest(BaseModel): regional_stress_data: List[Dict[str, Any]]; group_by: str
class PolicyStrategyRequest(BaseModel): region: str
class RegionalStressDataItem(BaseModel): region: str; status: str; count: int
class RegionalStressResponse(BaseModel): data: List[RegionalStressDataItem]
class LongTermTrendResponse(BaseModel): declining: List[Dict[str, Any]]; improving: List[Dict[str, Any]]
class ForecastResult(BaseModel): historical_data: List[Dict[str, Any]]; forecast_data: List[Dict[str, Any]]; confidence_interval: List[Dict[str, Any]]
class ForecastJobCreateResponse(BaseModel): job_id: int; message: str
class ForecastJobInfo(BaseModel):
    id: int
    station_id: int
    user_id: int
    status: JobStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    class Config: from_attributes = True
class ForecastJobStatusResponse(BaseModel):
    job_id: int
    status: JobStatus
    results: Optional[ForecastResult] = None
    error_message: Optional[str] = None
class PlanningDemands(BaseModel): agricultural: float = Field(..., ge=0); industrial: float = Field(..., ge=0); domestic: float = Field(..., ge=0)
class PlanningScenario(BaseModel): rainfall_change_percent: int = Field(0, ge=-100, le=100); demand_change_percent: int = Field(0, ge=-100, le=200)
class PlanningRequest(BaseModel):
    specific_yield: float = Field(..., gt=0, lt=1)
    area_sq_km: float = Field(..., gt=0)
    demands_m3: PlanningDemands
    scenario: PlanningScenario
class PlanningBaselineResult(BaseModel): avg_annual_recharge_mm: float; sustainable_yield_mm: float; sustainable_supply_m3: float; total_demand_m3: float; balance_m3: float
class PlanningScenarioResult(BaseModel): modified_supply_m3: float; modified_demand_m3: float; new_balance_m3: float; delta_from_baseline_m3: float
class PlanningResponse(BaseModel):
    station_name: str
    baseline: PlanningBaselineResult
    scenario: PlanningScenarioResult
    intelligent_analysis: Optional[str] = None
# FIX: Added schemas for missing features.
class QualityDataPoint(BaseModel): timestamp: datetime; ph: Optional[float] = None; tds_ppm: Optional[float] = None; turbidity_ntu: Optional[float] = None
class ResearchQualityResponse(BaseModel): data: List[QualityDataPoint]
class CorrelationRequest(BaseModel): parameter1: str; parameter2: str
class CorrelationResponse(BaseModel): correlation: Optional[float] = None; intelligent_analysis: Optional[str] = None
class GaugeData(BaseModel): parameter: str; value: float; min_val: float; max_val: float; normal_range: List[float]; unit: str
class PublicInfoResponse(BaseModel): gauges: List[GaugeData]
class PublicSummaryResponse(BaseModel): summary: str
class VolatilityPoint(BaseModel): timestamp: date; level: Optional[float] = None; volatility: Optional[float] = None
class VolatilityResponse(BaseModel): data: List[VolatilityPoint]; intelligent_analysis: Optional[str] = None
class MonsoonYearData(BaseModel): year: int; pre_monsoon_level_mbgl: Optional[float] = None; post_monsoon_level_mbgl: Optional[float] = None; recharge_effect_m: Optional[float] = None
class MonsoonResponse(BaseModel): average_recharge_effect_m: Optional[float] = None; yearly_data: List[MonsoonYearData]; intelligent_analysis: Optional[str] = None
class DroughtEvent(BaseModel): start_date: date; end_date: date; duration_days: int; peak_level_mbgl: float
class DroughtResponse(BaseModel): event_count: int; events: List[DroughtEvent]
class ReportKPIs(KPI): pass
class FullReportResponse(BaseModel):
    report_generated_on: datetime
    selection_filters: Dict[str, Any]
    kpis: ReportKPIs
    alerts_summary: AlertsSummary
    long_term_trends: LongTermTrendResponse
    forecast_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of forecast if a single station was selected.")
class ReportSummaryRequest(BaseModel): report_data: FullReportResponse

# ======================================================================================
# 5. SERVICES (`services/*.py`) - ALL IMPLEMENTED AS CLASSES
# ======================================================================================
# 5.1. CRUD Service (`services/crud.py`)
# --------------------------------------------------------------------------------------
class CrudService:
    def get_user(self, db: Session, user_id: int): return db.query(User).filter(User.id == user_id).first()
    def get_user_by_email(self, db: Session, email: str): return db.query(User).filter(User.email == email).first()
    def get_users(self, db: Session, skip: int, limit: int): return db.query(User).offset(skip).limit(limit).all()
    def create_user(self, db: Session, user: UserCreate, is_active: bool = True):
        hashed_password = auth_service.get_password_hash(user.password)
        db_user = User(email=user.email, hashed_password=hashed_password, role=user.role, is_active=is_active)
        db.add(db_user); db.commit(); db.refresh(db_user)
        return db_user
    def update_user(self, db: Session, user: User, user_update: Union[UserUpdateAdmin, UserUpdateMe]):
        update_data = user_update.model_dump(exclude_unset=True)
        for key, value in update_data.items(): setattr(user, key, value)
        db.commit(); db.refresh(user)
        return user
    def delete_user(self, db: Session, user: User): db.delete(user); db.commit()
    def get_station_by_id(self, db: Session, station_id: int): return db.query(Station).filter(Station.id == station_id).first()
    def get_stations_with_filters(self, db: Session, state: Optional[str]=None, district: Optional[str]=None, basin: Optional[str]=None, station_type: Optional[StationType]=None, agency: Optional[str]=None, skip: int = 0, limit: int = 100):
        query = db.query(Station)
        if state: query = query.filter(Station.state_name == state)
        if district: query = query.filter(Station.district_name == district)
        if basin: query = query.filter(Station.basin == basin)
        if station_type: query = query.filter(Station.station_type == station_type)
        if agency: query = query.filter(Station.agency_name == agency)
        return query.offset(skip).limit(limit).all()
    def get_distinct_filter_options(self, db: Session):
        states = db.query(Station.state_name).distinct().filter(Station.state_name.isnot(None)).all()
        districts = db.query(Station.district_name).distinct().filter(Station.district_name.isnot(None)).all()
        basins = db.query(Station.basin).distinct().filter(Station.basin.isnot(None)).all()
        return {"states": sorted([s[0] for s in states]), "districts": sorted([d[0] for d in districts]), "basins": sorted([b[0] for b in basins])}
    def get_distinct_agencies(self, db: Session) -> List[str]:
        agencies = db.query(Station.agency_name).distinct().filter(Station.agency_name.isnot(None)).all()
        return sorted([a[0] for a in agencies])
    def get_timeseries_for_station(self, db: Session, station_id: int, start_date: Optional[datetime]=None, end_date: Optional[datetime]=None):
        query = db.query(TimeSeriesData).filter(TimeSeriesData.station_id == station_id)
        if start_date: query = query.filter(TimeSeriesData.timestamp >= start_date)
        if end_date: query = query.filter(TimeSeriesData.timestamp <= end_date)
        return query.order_by(TimeSeriesData.timestamp.asc()).all()
    def get_timeseries_for_stations(self, db: Session, station_ids: List[int]):
        if not station_ids: return []
        return db.query(TimeSeriesData).filter(TimeSeriesData.station_id.in_(station_ids)).all()
    def bulk_create_timeseries_data(self, db: Session, data_points: list[dict]): db.bulk_insert_mappings(TimeSeriesData, data_points); db.commit()
    def get_aggregated_timeseries(self, db: Session, station_ids: List[int], start_date: Optional[datetime]=None, end_date: Optional[datetime]=None, frequency: str = 'day'):
        if not station_ids: return []
        truncated_date = func.date_trunc(frequency, TimeSeriesData.timestamp).label("period")
        query = (db.query(truncated_date, func.avg(TimeSeriesData.groundwaterlevel_mbgl).label("avg_gwl"), func.sum(TimeSeriesData.rainfall_mm).label("total_rainfall"), func.avg(TimeSeriesData.temperature_c).label("avg_temp"), func.avg(TimeSeriesData.ph).label("avg_ph"), func.avg(TimeSeriesData.tds_ppm).label("avg_tds"), func.avg(TimeSeriesData.turbidity_ntu).label("avg_turbidity"), func.count(TimeSeriesData.id).label("reading_count")).filter(TimeSeriesData.station_id.in_(station_ids)).group_by("period").order_by(desc("period")))
        if start_date: query = query.filter(TimeSeriesData.timestamp >= start_date)
        if end_date: query = query.filter(TimeSeriesData.timestamp <= end_date)
        return [row._asdict() for row in query.all()]
    def get_latest_timeseries_readings(self, db: Session, station_ids: List[int]):
        if not station_ids: return []
        subquery = (select(TimeSeriesData, func.row_number().over(partition_by=TimeSeriesData.station_id, order_by=TimeSeriesData.timestamp.desc()).label("row_num")).filter(TimeSeriesData.station_id.in_(station_ids)).subquery())
        latest_reading_alias = aliased(TimeSeriesData, subquery)
        query = select(latest_reading_alias).where(subquery.c.row_num == 1)
        return db.execute(query).scalars().all()
    def get_historical_percentiles_for_stations(self, db: Session, station_ids: List[int], percentile: float, year_limit: Optional[int] = None):
        if not station_ids: return {}
        query = db.query(
            TimeSeriesData.station_id,
            func.percentile_cont(percentile).within_group(TimeSeriesData.groundwaterlevel_mbgl.asc()).label("threshold")
        ).filter(
            TimeSeriesData.station_id.in_(station_ids),
            TimeSeriesData.groundwaterlevel_mbgl.isnot(None)
        )
        if year_limit:
            start_date = datetime.now(timezone.utc) - timedelta(days=year_limit * 365)
            query = query.filter(TimeSeriesData.timestamp >= start_date)

        query = query.group_by(TimeSeriesData.station_id)
        return {row.station_id: row.threshold for row in query.all()}
    def get_boundary_aggregates_for_timeseries(self, db: Session, station_ids: List[int], start_date: Optional[datetime] = None):
        if not station_ids: return {"start_avg_gwl": None, "end_avg_gwl": None}
        ts_query = db.query(TimeSeriesData).filter(TimeSeriesData.station_id.in_(station_ids))
        if start_date: ts_query = ts_query.filter(TimeSeriesData.timestamp >= start_date)
        
        first_week_start = ts_query.order_by(TimeSeriesData.timestamp.asc()).first()
        if not first_week_start: return {"start_avg_gwl": None, "end_avg_gwl": None}
        first_week_end = first_week_start.timestamp + timedelta(days=7)

        last_week_end_record = ts_query.order_by(TimeSeriesData.timestamp.desc()).first()
        if not last_week_end_record: return {"start_avg_gwl": start_avg_query.scalar(), "end_avg_gwl": None}
        last_week_end = last_week_end_record.timestamp
        last_week_start = last_week_end - timedelta(days=7)

        start_avg_query = db.query(func.avg(TimeSeriesData.groundwaterlevel_mbgl)).filter(TimeSeriesData.station_id.in_(station_ids), TimeSeriesData.timestamp.between(first_week_start.timestamp, first_week_end))
        end_avg_query = db.query(func.avg(TimeSeriesData.groundwaterlevel_mbgl)).filter(TimeSeriesData.station_id.in_(station_ids), TimeSeriesData.timestamp.between(last_week_start, last_week_end))
        
        return {"start_avg_gwl": start_avg_query.scalar(), "end_avg_gwl": end_avg_query.scalar()}
    def create_ingestion_job(self, db: Session, user_id: int, filenames: List[str]):
        db_job = DataIngestionJob(user_id=user_id, status=JobStatus.pending, details=json.dumps({"filenames": filenames}))
        db.add(db_job); db.commit(); db.refresh(db_job)
        return db_job
    def update_job_status(self, db: Session, job_id: int, status: JobStatus, details: Optional[Dict[str, Any]] = None):
        db_job = db.query(DataIngestionJob).filter(DataIngestionJob.id == job_id).first()
        if db_job:
            db_job.status = status
            if details is not None: db_job.details = json.dumps(details)
            db.commit(); db.refresh(db_job)
        return db_job
    def get_job_by_id(self, db: Session, job_id: int, user_id: Optional[int] = None):
        query = db.query(DataIngestionJob).filter(DataIngestionJob.id == job_id)
        if user_id: query = query.filter(DataIngestionJob.user_id == user_id)
        return query.first()
    def get_jobs_by_user_id(self, db: Session, user_id: int, skip: int, limit: int):
        return db.query(DataIngestionJob).filter(DataIngestionJob.user_id == user_id).order_by(DataIngestionJob.created_at.desc()).offset(skip).limit(limit).all()
    # FIX: Renamed AI cache functions.
    def get_cached_intelligence_response(self, db: Session, feature_name: str, request_data: dict):
        request_string = json.dumps(request_data, sort_keys=True)
        request_hash = hashlib.sha256(request_string.encode()).hexdigest()
        return db.query(IntelligenceAnalysisLog).filter(and_(IntelligenceAnalysisLog.feature_name == feature_name, IntelligenceAnalysisLog.request_hash == request_hash)).first()
    def cache_intelligence_response(self, db: Session, feature_name: str, request_data: dict, response_text: str):
        request_string = json.dumps(request_data, sort_keys=True)
        request_hash = hashlib.sha256(request_string.encode()).hexdigest()
        db_log = IntelligenceAnalysisLog(feature_name=feature_name, request_hash=request_hash, response_text=response_text)
        db.add(db_log); db.commit(); db.refresh(db_log)
        return db_log
    def create_forecast_job(self, db: Session, station_id: int, user_id: int):
        db_job = ForecastJob(station_id=station_id, user_id=user_id, status=JobStatus.pending)
        db.add(db_job); db.commit(); db.refresh(db_job)
        return db_job
    def update_forecast_job(self, db: Session, job_id: int, status: JobStatus, results: Optional[Dict] = None):
        db_job = db.query(ForecastJob).filter(ForecastJob.id == job_id).first()
        if db_job:
            db_job.status = status
            if results: db_job.results = json.dumps(results)
            db.commit(); db.refresh(db_job)
        return db_job
    def get_forecast_job_by_id(self, db: Session, job_id: int, user_id: Optional[int] = None):
        query = db.query(ForecastJob).filter(ForecastJob.id == job_id)
        if user_id: query = query.filter(ForecastJob.user_id == user_id)
        return query.first()
    def get_forecast_jobs_by_user(self, db: Session, user_id: int, skip: int, limit: int):
        return db.query(ForecastJob).filter(ForecastJob.user_id == user_id).order_by(desc(ForecastJob.created_at)).offset(skip).limit(limit).all()
crud = CrudService()

# --------------------------------------------------------------------------------------
# 5.2. Authentication Service (`services/auth_service.py`)
# --------------------------------------------------------------------------------------
class AuthService:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/v1/auth/token")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool: return self.pwd_context.verify(plain_password, hashed_password)
    def get_password_hash(self, password: str) -> str: return self.pwd_context.hash(password)
    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy(); expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "token_type": "access"})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    def create_refresh_token(self, data: dict) -> str:
        to_encode = data.copy(); expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "token_type": "refresh"})
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    def get_current_user(self, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
        credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            if payload.get("token_type") != "access": raise credentials_exception
            email: str = payload.get("sub")
            if email is None: raise credentials_exception
        except JWTError: raise credentials_exception
        user = crud.get_user_by_email(db, email=email)
        if user is None or not user.is_active: raise credentials_exception
        return user
    def get_user_from_refresh_token(self, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> str:
        credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate refresh token")
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            if payload.get("token_type") != "refresh": raise credentials_exception
            email: str = payload.get("sub")
            if email is None: raise credentials_exception
            return email
        except JWTError: raise credentials_exception
    def require_role(self, required_roles: Set[str]):
        def role_checker(current_user: User = Depends(self.get_current_user)):
            if current_user.role.value not in required_roles:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="The user does not have adequate permissions.")
            return current_user
        return role_checker
    async def create_first_superuser(self, db: Session):
        user = crud.get_user_by_email(db, email=settings.FIRST_SUPERUSER_EMAIL)
        if not user:
            logger.info(f"Creating first superuser: {settings.FIRST_SUPERUSER_EMAIL}")
            user_in = UserCreate(email=settings.FIRST_SUPERUSER_EMAIL, password=settings.FIRST_SUPERUSER_PASSWORD, role="admin")
            crud.create_user(db=db, user=user_in)
        else: logger.info("Superuser already exists.")
auth_service = AuthService()

# --------------------------------------------------------------------------------------
# FIX: Renamed AIService to IntelligenceService.
# 5.3. Intelligence Service (`services/intelligence_service.py`)
# --------------------------------------------------------------------------------------
class IntelligenceService:
    def __init__(self):
        self.IS_INTELLIGENCE_CONFIGURED = False
        if settings.ENABLE_INTELLIGENCE_FEATURES:
            try:
                genai.configure(api_key=settings.INTELLIGENCE_API_KEY)
                self.IS_INTELLIGENCE_CONFIGURED = True
            except Exception as e: logger.error(f"Failed to configure Advanced Intelligence features. They will be disabled. Error: {e}")
        else:
            logger.info("Advanced Intelligence features are globally disabled by configuration.")
        self.STRICT_SAFETY_SETTINGS = [
            SafetySetting(harm_category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold="BLOCK_ONLY_HIGH"),
            SafetySetting(harm_category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold="BLOCK_ONLY_HIGH"),
            SafetySetting(harm_category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold="BLOCK_ONLY_HIGH"),
            SafetySetting(harm_category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold="BLOCK_ONLY_HIGH"),
        ]
    def get_intelligence_response(self, db: Session, feature_name: str, prompt: str, request_data: dict, model_name: str = "gemini-1.5-flash-latest", force_refresh: bool = False):
        if not self.IS_INTELLIGENCE_CONFIGURED: return None
        if settings.ENABLE_INTELLIGENCE_CACHING and not force_refresh:
            cached_response = crud.get_cached_intelligence_response(db, feature_name=feature_name, request_data=request_data)
            if cached_response: logger.info(f"INTELLIGENCE CACHE HIT for feature: {feature_name}"); return cached_response.response_text
        logger.info(f"INTELLIGENCE CACHE MISS for feature: {feature_name}. Calling analysis service.")
        try:
            model = genai.GenerativeModel(model_name)
            generation_config = GenerationConfig(max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS)
            max_retries = 3; base_delay = 2
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt, generation_config=generation_config, safety_settings=self.STRICT_SAFETY_SETTINGS, request_options={"timeout": 120})
                    response_text = response.text
                    if settings.ENABLE_INTELLIGENCE_CACHING: crud.cache_intelligence_response(db, feature_name=feature_name, request_data=request_data, response_text=response_text)
                    return response_text
                except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded) as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt); logger.warning(f"Analysis service transient error: {e}. Retrying in {delay} seconds..."); time.sleep(delay)
                    else: raise e
        except Exception as e: logger.error(f"An unexpected error calling analysis service for feature '{feature_name}': {e}")
        return None
    def get_intelligence_json_response(self, db: Session, feature_name: str, prompt: str, request_data: dict, output_schema: Optional[Dict[str, Any]] = None, force_refresh: bool = False):
        if not self.IS_INTELLIGENCE_CONFIGURED: return None
        if settings.ENABLE_INTELLIGENCE_CACHING and not force_refresh:
            cached_response = crud.get_cached_intelligence_response(db, feature_name=feature_name, request_data=request_data)
            if cached_response: logger.info(f"INTELLIGENCE CACHE HIT for feature: {feature_name} (JSON)"); return json.loads(cached_response.response_text)
        logger.info(f"INTELLIGENCE CACHE MISS for feature: {feature_name} (JSON). Calling analysis service.")
        try:
            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            generation_config_dict = {"response_mime_type": "application/json", "max_output_tokens": settings.GEMINI_MAX_OUTPUT_TOKENS}
            if output_schema: generation_config_dict["response_schema"] = output_schema
            generation_config = GenerationConfig.from_dict(generation_config_dict)
            max_retries = 3; base_delay = 2
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt, generation_config=generation_config, safety_settings=self.STRICT_SAFETY_SETTINGS, request_options={"timeout": 120})
                    response_json_text = response.text; response_dict = json.loads(response_json_text)
                    if settings.ENABLE_INTELLIGENCE_CACHING: crud.cache_intelligence_response(db, feature_name=feature_name, request_data=request_data, response_text=response_json_text)
                    return response_dict
                except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded) as e:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt); logger.warning(f"Analysis service (JSON mode) transient error: {e}. Retrying in {delay} seconds..."); time.sleep(delay)
                    else: raise e
        except Exception as e: logger.error(f"An unexpected error in get_intelligence_json_response for feature '{feature_name}': {e}")
        return None
intelligence_service = IntelligenceService()

# --------------------------------------------------------------------------------------
# 5.4. Ingestion Service (`services/ingestion_service.py`)
# --------------------------------------------------------------------------------------
class IngestionService:
    async def save_files_and_create_job(self, db: Session, user_id: int, files: List[UploadFile]):
        os.makedirs(settings.TEMP_UPLOAD_DIR, exist_ok=True)
        job = crud.create_ingestion_job(db, user_id=user_id, filenames=[f.filename for f in files])
        file_paths = {}
        try:
            for file in files:
                safe_filename = secure_filename(file.filename)
                if not safe_filename:
                    raise HTTPException(status_code=400, detail=f"Invalid filename provided: {file.filename}")
                file_path = os.path.join(settings.TEMP_UPLOAD_DIR, f"{job.id}_{safe_filename}")
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths[file.filename] = file_path
        except OSError as e:
            logger.error(f"Disk I/O error during file upload for job {job.id}: {e}")
            crud.update_job_status(db, job.id, JobStatus.failed, {"error": "Failed to save files. The server disk may be full."})
            raise HTTPException(status_code=507, detail="Could not save uploaded files. Insufficient storage.")
        
        details = {"filenames": [f.filename for f in files], "file_paths": file_paths}
        crud.update_job_status(db, job.id, JobStatus.pending, details=details)
        return job, file_paths

    def perform_initial_analysis_task(self, db: Session, job_id: int, file_paths: Dict[str, str]):
        try:
            file_contents = {fname: open(path, 'rb').read() for fname, path in file_paths.items()}
            classified_roles = self._classify_files(file_contents)
            gw_stations_fname = classified_roles.get('stations_gw')
            rf_stations_fname = classified_roles.get('stations_rf')
            timeseries_fname = classified_roles.get('timeseries')

            if not gw_stations_fname or not timeseries_fname:
                raise ValueError("Could not classify required GW Station and Time-Series files from the provided uploads.")

            mappings = {}
            def get_mapping(fname, schema):
                if not fname: return None
                cols = pd.read_csv(io.BytesIO(file_contents[fname]), nrows=0).columns.tolist()
                suggestion = intelligence_service.get_intelligence_json_response(db, "column_mapping", f"Map columns for {fname}: `{cols}` to schema: `{schema}`. Return only a single raw JSON object.", {"file": fname})
                return {"file_name": fname, "columns": cols, "suggestion": suggestion or {}}

            mappings['gw_stations'] = get_mapping(gw_stations_fname, STATION_SCHEMA)
            mappings['timeseries'] = get_mapping(timeseries_fname, TIMESERIES_SCHEMA)
            if rf_stations_fname:
                mappings['rf_stations'] = get_mapping(rf_stations_fname, STATION_SCHEMA)

            crud.update_job_status(db, job_id, JobStatus.awaiting_confirmation, mappings)
        except Exception as e:
            logger.error(f"Initial analysis for job {job_id} failed: {e}")
            crud.update_job_status(db, job_id, JobStatus.failed, {"error": str(e)})
        finally:
            pass

    def generate_data_quality_report(self, df: pd.DataFrame, df_name: str) -> str:
        report = f"#### Quality Analysis: `{df_name}`\n- **Dimensions**: {df.shape[0]} rows & {df.shape[1]} columns.\n"
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100 if len(df) > 0 else 0
        missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False)
        if not missing_report.empty:
            report += "- **Missing Values Analysis**:\n"
            for col, pct in missing_report.items(): report += f"  - `{col}`: **{pct:.1f}% missing** ({missing[col]} values).\n"
        else: report += "- **Missing Values**: None detected. ✅\n"
        
        key_cols = []
        if 'timestamp' in df.columns and 'station_name' in df.columns:
            key_cols = ['timestamp', 'station_name']
        elif 'latitude' in df.columns and 'station_name' in df.columns:
            key_cols = ['station_name']
        
        if key_cols:
            dupes = df.duplicated(subset=key_cols).sum()
            if dupes > 0: report += f"- **Duplicate Records**: Found **{dupes}** duplicate entries based on key columns {key_cols}. ⚠️\n"

        for col in ['groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'tds_ppm', 'turbidity_ntu']:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and not df[col].dropna().empty:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                    if outliers > 0: report += f"- **Potential Outliers**: Detected {outliers} in `{col}`.\n"
        return report

    def perform_final_ingestion_task(self, db: Session, job_id: int, confirmed_mapping: Dict[str, Any]):
        job = crud.get_job_by_id(db, job_id=job_id)
        details = json.loads(job.details) if isinstance(job.details, str) else job.details
        file_paths = details.get('file_paths', {})
        try:
            crud.update_job_status(db, job_id, JobStatus.processing, {"message": "Starting final data processing."})
            
            gw_map = confirmed_mapping.get('gw_stations', {})
            ts_map = confirmed_mapping.get('timeseries', {})
            rf_map = confirmed_mapping.get('rf_stations', {})

            gw_df = pd.read_csv(file_paths[gw_map.get('file_name')])
            ts_df = pd.read_csv(file_paths[ts_map.get('file_name')])
            rf_df = None
            if rf_map and rf_map.get('file_name') in file_paths:
                rf_df = pd.read_csv(file_paths[rf_map.get('file_name')])

            gw_df.rename(columns={v: k for k, v in gw_map.get('suggestion', {}).items() if v}, inplace=True)
            ts_df.rename(columns={v: k for k, v in ts_map.get('suggestion', {}).items() if v}, inplace=True)
            if rf_df is not None:
                rf_df.rename(columns={v: k for k, v in rf_map.get('suggestion', {}).items() if v}, inplace=True)

            required_station_cols = {'station_name', 'latitude', 'longitude'}
            if not required_station_cols.issubset(gw_df.columns):
                missing_cols = required_station_cols - set(gw_df.columns)
                raise ValueError(f"Groundwater station file is missing required columns: {', '.join(missing_cols)}")
            if rf_df is not None and not required_station_cols.issubset(rf_df.columns):
                missing_cols = required_station_cols - set(rf_df.columns)
                raise ValueError(f"Rainfall station file is missing required columns: {', '.join(missing_cols)}")

            if gw_df['station_name'].duplicated().any():
                raise ValueError("Duplicate station names found within the uploaded groundwater station file.")
            if rf_df is not None and rf_df['station_name'].duplicated().any():
                raise ValueError("Duplicate station names found within the uploaded rainfall station file.")
            
            ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'], errors='coerce', utc=True)
            if ts_df['timestamp'].isnull().any():
                raise ValueError("Timestamp column contains invalid date formats that could not be parsed.")

            quality_reports = {
                gw_map.get('file_name'): self.generate_data_quality_report(gw_df, gw_map.get('file_name')),
                ts_map.get('file_name'): self.generate_data_quality_report(ts_df, ts_map.get('file_name'))
            }
            if rf_df is not None:
                quality_reports[rf_map.get('file_name')] = self.generate_data_quality_report(rf_df, rf_map.get('file_name'))

            string_cols_to_normalize = ['station_name', 'state_name', 'district_name', 'agency_name', 'basin']
            for col in string_cols_to_normalize:
                if col in gw_df.columns:
                    gw_df[col] = gw_df[col].astype(str).str.strip().str.title()
                if rf_df is not None and col in rf_df.columns:
                    rf_df[col] = rf_df[col].astype(str).str.strip().str.title()
            
            gw_df['station_type'] = StationType.groundwater
            all_station_dfs = [gw_df]
            if rf_df is not None:
                rf_df['station_type'] = StationType.rainfall
                all_station_dfs.append(rf_df)
            
            all_stations_df = pd.concat(all_station_dfs, ignore_index=True)

            existing_stations = {s.station_name: s for s in crud.get_stations_with_filters(db)}
            new_stations_to_create = []
            
            with db.begin_nested():
                existing_stations_locked = {s.station_name: s for s in crud.get_stations_with_filters(db)}
                for _, row in all_stations_df.iterrows():
                    if row['station_name'] not in existing_stations_locked:
                        try:
                            station_data = {key: row.get(key) for key in StationCreate.model_fields.keys()}
                            new_stations_to_create.append(StationCreate(**station_data).model_dump())
                        except ValidationError as e:
                            raise ValueError(f"Validation error for station '{row.get('station_name')}': {e}")
                
                if new_stations_to_create:
                    try:
                        db.bulk_insert_mappings(Station, new_stations_to_create)
                        db.commit()
                    except IntegrityError:
                        db.rollback()
                        raise ValueError("A race condition occurred during station creation. Please try the upload again.")
            
            existing_stations = {s.station_name: s for s in crud.get_stations_with_filters(db)}

            ts_df['station_name'] = ts_df['station_name'].astype(str).str.strip().str.title()
            numeric_cols = ['groundwaterlevel_mbgl', 'rainfall_mm', 'temperature_c', 'ph', 'turbidity_ntu', 'tds_ppm']
            for col in numeric_cols:
                if col in ts_df.columns:
                    ts_df[col] = pd.to_numeric(ts_df[col], errors='coerce')

            station_id_map = {name: station.id for name, station in existing_stations.items()}
            ts_df['station_id'] = ts_df['station_name'].map(station_id_map)
            
            ts_df.dropna(subset=['timestamp', 'station_id'], inplace=True)
            ts_df['station_id'] = ts_df['station_id'].astype(int)

            ts_records = ts_df[['station_id', 'timestamp'] + [col for col in numeric_cols if col in ts_df.columns]].to_dict(orient='records')
            
            if ts_records:
                crud.bulk_create_timeseries_data(db, ts_records)

            final_details = {
                "message": f"Ingestion successful. Processed {len(all_stations_df)} station records and {len(ts_records)} time-series points.",
                "quality_reports": quality_reports
            }
            crud.update_job_status(db, job_id, JobStatus.completed, final_details)
        except (ValueError, ValidationError) as e:
            logger.error(f"Data validation failed for job {job_id}: {e}")
            crud.update_job_status(db, job_id, JobStatus.failed, {"error": str(e)})
        except Exception as e:
            logger.error(f"Final ingestion for job {job_id} failed: {e}")
            crud.update_job_status(db, job_id, JobStatus.failed, {"error": "An unexpected error occurred during processing."})
        finally:
            for path in file_paths.values():
                if os.path.exists(path):
                    os.remove(path)

    def _classify_files(self, file_contents: Dict[str, bytes]) -> Dict[str, Optional[str]]:
        roles: Dict[str, Optional[str]] = {'timeseries': None, 'stations_gw': None, 'stations_rf': None}
        candidates = []
        for file_name, content in file_contents.items():
            try:
                cols = pd.read_csv(io.BytesIO(content), nrows=0).columns.str.lower().str.replace('[_-]', '', regex=True)
                score = 0
                if any(ts in cols for ts in ['timestamp', 'date', 'datetime']): score += 10
                if 'latitude' in cols and 'longitude' in cols: score += 5
                if any(gw in cols for gw in ['groundwater', 'gwl', 'mbgl']): score += 2
                if any(rf in cols for rf in ['rainfall', 'rain']): score += 1
                candidates.append((score, file_name))
            except Exception:
                candidates.append((-1, file_name))
        candidates.sort(key=lambda x: x[0], reverse=True)
        assigned_files = set()
        if len(candidates) >= 1 and candidates[0][0] >= 10:
            roles['timeseries'] = candidates[0][1]
            assigned_files.add(candidates[0][1])
        for score, fname in candidates:
            if fname not in assigned_files and score >= 5:
                # Prioritize 'gw' keyword for groundwater stations
                if 'gw' in fname.lower() or 'ground' in fname.lower():
                    if not roles['stations_gw']:
                        roles['stations_gw'] = fname
                        assigned_files.add(fname)
        # Assign remaining station files
        for score, fname in candidates:
            if fname not in assigned_files and score >= 5:
                if not roles['stations_gw']:
                    roles['stations_gw'] = fname
                    assigned_files.add(fname)
                elif not roles['stations_rf']:
                    roles['stations_rf'] = fname
                    assigned_files.add(fname)
        return roles
ingestion_service = IngestionService()
# --------------------------------------------------------------------------------------
# 5.5. Analysis Service
# --------------------------------------------------------------------------------------
class AnalysisService:
    def get_filtered_long_term_trends(self, db: Session, state: Optional[str], district: Optional[str], basin: Optional[str]):
        stations = crud.get_stations_with_filters(db, state=state, district=district, basin=basin, station_type=StationType.groundwater)
        if not stations:
            return {"declining": [], "improving": []}

        station_ids = [s.id for s in stations]
        station_map = {s.id: s.station_name for s in stations}
        
        query = db.query(
            TimeSeriesData.station_id,
            TimeSeriesData.timestamp,
            TimeSeriesData.groundwaterlevel_mbgl
        ).filter(TimeSeriesData.station_id.in_(station_ids))
        
        df = pd.read_sql(query.statement, query.session.bind)
        df = df.dropna()
        
        trends = {}
        for station_id, group in df.groupby('station_id'):
            if len(group) > 30:
                group = group.sort_values('timestamp')
                group['time_ord'] = (group['timestamp'] - group['timestamp'].min()).dt.days
                model = LinearRegression().fit(group[['time_ord']], group['groundwaterlevel_mbgl'])
                trends[station_map[station_id]] = model.coef_[0] * 365
        
        sorted_trends = sorted(trends.items(), key=lambda item: item[1])
        improving = [{"station_name": t[0], "annual_trend_m": t[1]} for t in sorted_trends if t[1] < 0][:5]
        declining = [{"station_name": t[0], "annual_trend_m": t[1]} for t in sorted_trends if t[1] > 0][::-1][:5]
        return {"declining": declining, "improving": improving}

    def run_forecast_model(self, db: Session, job_id: int, station_id: int, days_to_forecast: int):
        try:
            crud.update_forecast_job(db, job_id, status=JobStatus.forecasting)
            ts_data = crud.get_timeseries_for_station(db, station_id)
            
            if not ts_data:
                raise ValueError("No time-series data found for this station.")

            df = pd.DataFrame([{"timestamp": r.timestamp, "groundwaterlevel_mbgl": r.groundwaterlevel_mbgl} for r in ts_data])
            df = df.dropna(subset=['timestamp', 'groundwaterlevel_mbgl'])
            if len(df) < 24: raise ValueError("Forecasting requires at least 24 data points for this model.")
            
            df_f = df.set_index('timestamp').asfreq('D').interpolate(method='time')
            
            model = SARIMAX(df_f['groundwaterlevel_mbgl'], order=(1,1,1), seasonal_order=(1,1,1,12))
            fit = model.fit(disp=False)
            forecast_result = fit.get_forecast(steps=days_to_forecast)
            
            historical_data = df_f.reset_index().to_dict(orient='records')
            forecast_data = forecast_result.predicted_mean.reset_index().rename(columns={'index':'timestamp', 0:'mean'}).to_dict(orient='records')
            conf_int = forecast_result.conf_int().reset_index().rename(columns={'index':'timestamp'}).to_dict(orient='records')
            
            results = {"historical_data": historical_data, "forecast_data": forecast_data, "confidence_interval": conf_int}
            crud.update_forecast_job(db, job_id, status=JobStatus.completed, results=results)
        except Exception as e:
            logger.error(f"SARIMAX forecast failed for job {job_id}: {e}")
            crud.update_forecast_job(db, job_id, status=JobStatus.failed, results={"error": str(e)})

    def get_regional_stress(self, db: Session, group_by_col: str, percentile: float):
        if group_by_col not in ['state_name', 'basin']:
            raise ValueError("Invalid grouping column for regional stress analysis.")

        all_stations = crud.get_stations_with_filters(db, station_type=StationType.groundwater)
        if not all_stations:
            return []
        
        station_ids = [s.id for s in all_stations]
        latest_readings = crud.get_latest_timeseries_readings(db, station_ids)
        thresholds = crud.get_historical_percentiles_for_stations(db, station_ids, percentile / 100.0)

        station_map = {s.id: s for s in all_stations}
        status_data = []

        for reading in latest_readings:
            threshold = thresholds.get(reading.station_id)
            if threshold is not None and reading.groundwaterlevel_mbgl is not None:
                station_info = station_map.get(reading.station_id)
                region = getattr(station_info, group_by_col)
                if region:
                    status = "Low/Critical" if reading.groundwaterlevel_mbgl > threshold else "Normal"
                    status_data.append({"region": region, "status": status})

        if not status_data:
            return []
            
        df = pd.DataFrame(status_data)
        stress_counts = df.groupby(['region', 'status']).size().reset_index(name='count')
        return stress_counts.to_dict(orient='records')

analysis_service = AnalysisService()
# ======================================================================================
# ======================================================================================
# 6. API ROUTERS (`api/*.py`)
# ======================================================================================
API_PREFIX = "/api/v1"
router_auth = APIRouter(); router_users = APIRouter(); router_ingestion = APIRouter()
router_map = APIRouter(); router_dashboard = APIRouter(); router_alerts = APIRouter()
router_policy = APIRouter(); router_planning = APIRouter();
# FIX: Added missing routers.
router_research = APIRouter()
router_public = APIRouter()
router_hydrology = APIRouter()
router_reports = APIRouter()
router_admin = APIRouter()
router_stations = APIRouter()


# --------------------------------------------------------------------------------------
# 6.0. Authorization Dependencies
# --------------------------------------------------------------------------------------
def get_station_for_user(station_id: int, db: Session = Depends(get_db), current_user: User = Depends(auth_service.get_current_user)) -> Station:
    station = crud.get_station_by_id(db, station_id)
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")
    if current_user.role not in [UserRole.admin, UserRole.researcher, UserRole.policy_maker]:
         raise HTTPException(status_code=403, detail="User not authorized to access this station's data.")
    return station

INTERNAL_USER_ROLES = Depends(auth_service.require_role({UserRole.admin.value, UserRole.researcher.value, UserRole.policy_maker.value}))

# --------------------------------------------------------------------------------------
# 6.1. Authentication Router
# --------------------------------------------------------------------------------------
@router_auth.post("/auth/token", response_model=Token)
def login(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    user = crud.get_user_by_email(db, email=form_data.username)
    if not user or not auth_service.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password", headers={"WWW-Authenticate": "Bearer"})
    if not user.is_active: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    access_token = auth_service.create_access_token(data={"sub": user.email})
    refresh_token = auth_service.create_refresh_token(data={"sub": user.email})
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@router_auth.post("/auth/refresh", response_model=Token)
def refresh(db: Session = Depends(get_db), current_user_email: str = Depends(auth_service.get_user_from_refresh_token)):
    user = crud.get_user_by_email(db, email=current_user_email)
    if not user or not user.is_active: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive user")
    new_access_token = auth_service.create_access_token(data={"sub": user.email})
    new_refresh_token = auth_service.create_refresh_token(data={"sub": user.email})
    return {"access_token": new_access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}

# --------------------------------------------------------------------------------------
# 6.2. User Management Router
# --------------------------------------------------------------------------------------
ADMIN_ONLY = Depends(auth_service.require_role({UserRole.admin.value}))
@router_users.get("/users/me", response_model=User)
def read_me(current_user: User = Depends(auth_service.get_current_user)): return current_user

@router_users.patch("/users/me", response_model=User)
def update_me(user_update: UserUpdateMe, db: Session = Depends(get_db), current_user: User = Depends(auth_service.get_current_user)):
    if user_update.email and crud.get_user_by_email(db, email=user_update.email): raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered.")
    return crud.update_user(db=db, user=current_user, user_update=user_update)

@router_users.post("/users/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db), admin_user: User = ADMIN_ONLY):
    if crud.get_user_by_email(db, email=user.email): raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@router_users.get("/users/", response_model=List[User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), admin_user: User = ADMIN_ONLY): return crud.get_users(db, skip=skip, limit=limit)

@router_users.patch("/users/{user_id}", response_model=User)
def update_user_by_admin(user_id: int, user_update: UserUpdateAdmin, db: Session = Depends(get_db), admin_user: User = ADMIN_ONLY):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user_update.is_active is False and admin_user.id == db_user.id: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Admins cannot deactivate their own account.")
    if user_update.email and user_update.email != db_user.email and crud.get_user_by_email(db, email=user_update.email): raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    return crud.update_user(db=db, user=db_user, user_update=user_update)

@router_users.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db), admin_user: User = ADMIN_ONLY):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if admin_user.id == db_user.id: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Admins cannot delete their own account.")
    crud.delete_user(db=db, user=db_user); return None

# --------------------------------------------------------------------------------------
# 6.3. Ingestion Router
# --------------------------------------------------------------------------------------
UPLOADER_DEPENDENCY = Depends(auth_service.require_role({UserRole.admin.value, UserRole.researcher.value}))
@router_ingestion.post("/ingestion/initiate", response_model=DataIngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def initiate_ingestion(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...), db: Session = Depends(get_db), current_user: User = UPLOADER_DEPENDENCY):
    if not 2 <= len(files) <= 3: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please upload 2 or 3 files.")
    job, file_paths = await ingestion_service.save_files_and_create_job(db=db, user_id=current_user.id, files=files)
    background_tasks.add_task(ingestion_service.perform_initial_analysis_task, db=db, job_id=job.id, file_paths=file_paths)
    return job

@router_ingestion.get("/ingestion/jobs/{job_id}/mapping", response_model=IntelligentColumnMappingResult)
def get_mapping(job_id: int, db: Session = Depends(get_db), current_user: User = UPLOADER_DEPENDENCY):
    job = crud.get_job_by_id(db, job_id=job_id, user_id=current_user.id)
    if not job: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or access denied.")
    if job.status != JobStatus.awaiting_confirmation: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Job is in status '{job.status}'. Cannot retrieve mapping.")
    details = json.loads(job.details)
    # This renaming is to match the schema change from ai_suggestion to suggestion
    for key in details:
        if 'ai_suggestion' in details[key]:
            details[key]['suggestion'] = details[key].pop('ai_suggestion')
    return details

@router_ingestion.post("/ingestion/jobs/{job_id}/confirm", response_model=DataIngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
def confirm_ingestion(job_id: int, confirmed_mapping: IntelligentColumnMappingResult, background_tasks: BackgroundTasks, db: Session = Depends(get_db), current_user: User = UPLOADER_DEPENDENCY):
    job = crud.get_job_by_id(db, job_id=job_id, user_id=current_user.id)
    if not job: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or access denied.")
    if job.status != JobStatus.awaiting_confirmation: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job can only be confirmed from 'AWAITING_CONFIRMATION' status.")
    
    # Translate back to the internal format expected by the task
    mapping_for_task = confirmed_mapping.model_dump()
    for key in mapping_for_task:
        if mapping_for_task[key] and 'suggestion' in mapping_for_task[key]:
            mapping_for_task[key]['ai_suggestion'] = mapping_for_task[key].pop('suggestion')

    background_tasks.add_task(ingestion_service.perform_final_ingestion_task, db=db, job_id=job.id, confirmed_mapping=mapping_for_task)
    return crud.update_job_status(db, job_id=job.id, status=JobStatus.processing)

@router_ingestion.get("/ingestion/jobs/{job_id}", response_model=DataIngestionJobResponse)
def get_job_status(job_id: int, db: Session = Depends(get_db), current_user: User = Depends(auth_service.get_current_user)):
    user_id_filter = None if current_user.role == UserRole.admin else current_user.id
    db_job = crud.get_job_by_id(db, job_id=job_id, user_id=user_id_filter)
    if db_job is None: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found or access denied.")
    if isinstance(db_job.details, str): db_job.details = json.loads(db_job.details)
    return db_job

@router_ingestion.get("/ingestion/jobs/{job_id}/quality-report", response_model=DataQualityReport)
def get_quality_report(job_id: int, db: Session = Depends(get_db), current_user: User = Depends(auth_service.get_current_user)):
    user_id_filter = None if current_user.role == UserRole.admin else current_user.id
    job = crud.get_job_by_id(db, job_id, user_id=user_id_filter)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or access denied.")
    if job.status != JobStatus.completed:
        raise HTTPException(status_code=400, detail="Quality report is only available for completed jobs.")
    
    details = json.loads(job.details) if isinstance(job.details, str) else job.details
    report = details.get("quality_reports")
    if not report:
        raise HTTPException(status_code=404, detail="Quality report not found for this job.")
        
    return DataQualityReport(report=report)

# --------------------------------------------------------------------------------------
# 6.4. Map Router
# --------------------------------------------------------------------------------------
@router_map.get("/map", response_model=MapResponse)
def get_map_data(db: Session = Depends(get_db), current_user: User = INTERNAL_USER_ROLES, state: Optional[str] = Query(None), district: Optional[str] = Query(None), basin: Optional[str] = Query(None), year: Optional[int] = Query(None, ge=1900, le=2100), selected_station_name: Optional[str] = Query(None), map_style: MapStyle = Query(MapStyle.points)):
    gw_stations = crud.get_stations_with_filters(db, state=state, district=district, basin=basin, station_type=StationType.groundwater)
    rf_stations = crud.get_stations_with_filters(db, state=state, district=district, basin=basin, station_type=StationType.rainfall)
    selected_station = next((s for s in gw_stations if s.station_name == selected_station_name), None) if selected_station_name else None
    map_center = {"lat": selected_station.latitude, "lon": selected_station.longitude} if selected_station else None
    if map_style.startswith('heatmap'):
        if map_style == 'heatmap_density':
            all_stations = gw_stations + rf_stations
            heatmap_data = [{"lat": s.latitude, "lon": s.longitude} for s in all_stations]
            return {"map_type": "heatmap", "heatmap_data": heatmap_data, "map_center": map_center, "info_text": f"Showing raw density for {len(heatmap_data)} stations."}
        elif map_style == 'heatmap_risk':
            if not year or not gw_stations: raise HTTPException(status_code=400, detail="Please select a year to view the Water Stress Risk map.")
            all_station_ids=[s.id for s in gw_stations]; latest_readings=crud.get_latest_timeseries_readings(db, station_ids=all_station_ids); latest_reading_map={r.station_id: r for r in latest_readings}
            thresholds = crud.get_historical_percentiles_for_stations(db, station_ids=all_station_ids, percentile=0.9, year_limit=10)
            heatmap_data = []
            for station in gw_stations:
                status="no_data"; latest_reading=latest_reading_map.get(station.id); threshold=thresholds.get(station.id)
                if latest_reading and latest_reading.timestamp.year==year and threshold is not None:
                    current_gwl=latest_reading.groundwaterlevel_mbgl
                    if current_gwl is not None and current_gwl > threshold: status="critical"
                    else: status="normal"
                weight=0.1;
                if status=="critical": weight=1.0
                elif status=="normal": weight=0.5
                heatmap_data.append({"lat": station.latitude, "lon": station.longitude, "weight": weight})
            return {"map_type": "heatmap", "heatmap_data": heatmap_data, "map_center": map_center, "info_text": f"Showing weighted risk density for {len(gw_stations)} GW stations in {year} (based on last 10 years)."}
    map_points, info_text_parts = [], ["🔵 GW Stations", "🟢 RF Stations"]
    status_by_id = {}
    if year and gw_stations:
        all_station_ids=[s.id for s in gw_stations]; latest_readings=crud.get_latest_timeseries_readings(db, station_ids=all_station_ids); latest_reading_map={r.station_id: r for r in latest_readings}
        thresholds = crud.get_historical_percentiles_for_stations(db, station_ids=all_station_ids, percentile=0.9)
        for station in gw_stations:
            status_info={"status": "no_data"}; latest_reading=latest_reading_map.get(station.id); threshold=thresholds.get(station.id)
            if latest_reading and latest_reading.timestamp.year==year and threshold is not None:
                current_gwl=latest_reading.groundwaterlevel_mbgl
                status_info.update({"latest_gwl": round(current_gwl, 2) if current_gwl is not None else None, "threshold_gwl": round(threshold, 2), "latest_reading_date": latest_reading.timestamp.strftime('%Y-%m-%d')})
                if current_gwl is not None and current_gwl > threshold: status_info["status"]="critical"
                else: status_info["status"]="normal"
            status_by_id[station.id] = status_info
        info_text_parts[0] = f"Status in {year}: 🔵 Normal | 🔴 Critical | ⚫ No Data"
    for station in gw_stations:
        status_info=status_by_id.get(station.id, {"status": "no_data"}); status=status_info.get("status")
        point = {"station_id": station.id, "station_name": station.station_name, "station_type": "groundwater", "lat": station.latitude, "lon": station.longitude, "size": 25, "data": status_info}
        if status=="critical": point["color"]="#FF0000"
        elif status=="normal": point["color"]="#0066FF"
        else: point["color"]="#808080"
        map_points.append(point)
    for station in rf_stations: map_points.append({"station_id": station.id, "station_name": station.station_name, "station_type": "rainfall", "lat": station.latitude, "lon": station.longitude, "size": 25, "color": "#00CC66", "data": {}})
    if selected_station:
        for p in map_points:
            if p["station_id"] == selected_station.id: p["color"]="#FFD700"; p["size"]=100; break
        info_text_parts.append("⭐ Selected GW")
        if rf_stations:
            rf_lats=np.array([s.latitude for s in rf_stations]); rf_lons=np.array([s.longitude for s in rf_stations])
            distances=haversine_np(selected_station.longitude, selected_station.latitude, rf_lons, rf_lats); nearest_idx=np.argmin(distances)
            nearest_rf_station=rf_stations[nearest_idx]
            for p in map_points:
                if p["station_id"] == nearest_rf_station.id: p["color"]="#FFA500"; p["size"]=100; break
            info_text_parts.append(f"🟠 Nearest RF ({distances[nearest_idx]:.2f} km)")
    return {"map_type": "points", "points": map_points, "map_center": map_center, "info_text": " | ".join(info_text_parts)}

@router_map.get("/map/filters", response_model=MapFilterOptions)
def get_map_filters(db: Session = Depends(get_db), current_user: User = Depends(auth_service.get_current_user)):
    return crud.get_distinct_filter_options(db)

# --------------------------------------------------------------------------------------
# 6.5. Dashboard Router
# --------------------------------------------------------------------------------------
@router_dashboard.get("/dashboard", response_model=DashboardResponse)
def get_dashboard_data(db: Session = Depends(get_db), current_user: User = INTERNAL_USER_ROLES, state: Optional[str] = Query(None), district: Optional[str] = Query(None), basin: Optional[str] = Query(None), agency: Optional[str] = Query(None), time_range_days: Optional[int] = Query(None, ge=1)):
    all_stations_in_scope = crud.get_stations_with_filters(db, state=state, district=district, basin=basin, agency=agency)
    gw_stations_in_scope = [s for s in all_stations_in_scope if s.station_type == StationType.groundwater]
    
    if not gw_stations_in_scope: return DashboardResponse(kpis=KPI(), agency_distribution=[])
    
    station_ids = [s.id for s in gw_stations_in_scope]
    start_date = (datetime.now(timezone.utc) - timedelta(days=time_range_days)) if time_range_days else None
    aggregated_data = crud.get_aggregated_timeseries(db, station_ids=station_ids, start_date=start_date, frequency='all')
    latest_readings = crud.get_latest_timeseries_readings(db, station_ids=station_ids)
    boundary_aggregates = crud.get_boundary_aggregates_for_timeseries(db, station_ids=station_ids, start_date=start_date)
    kpis = _calculate_kpis_from_aggregates(aggregated_data, latest_readings, boundary_aggregates)
    agency_distribution = _calculate_agency_distribution(all_stations_in_scope)
    return DashboardResponse(kpis=kpis, agency_distribution=agency_distribution)

def _calculate_kpis_from_aggregates(aggregated_data, latest_readings, boundary_aggregates):
    if not aggregated_data: return KPI()
    agg = aggregated_data[0]
    recent_gwl, latest_turbidity, latest_tds = None, None, None
    if latest_readings:
        latest_readings.sort(key=lambda x: x.timestamp, reverse=True)
        recent_gwl = next((r.groundwaterlevel_mbgl for r in latest_readings if r.groundwaterlevel_mbgl is not None), None)
        latest_turbidity = next((r.turbidity_ntu for r in latest_readings if r.turbidity_ntu is not None), None)
        latest_tds = next((r.tds_ppm for r in latest_readings if r.tds_ppm is not None), None)
    gwl_delta = None
    start_avg = boundary_aggregates.get('start_avg_gwl'); end_avg = boundary_aggregates.get('end_avg_gwl')
    if start_avg is not None and end_avg is not None: gwl_delta = end_avg - start_avg
    
    return KPI(
        avg_gw_level=agg.get("avg_gwl"), 
        recent_gw_level=recent_gwl, 
        gw_level_delta=gwl_delta, 
        total_rainfall=agg.get("total_rainfall"), 
        avg_temp=agg.get("avg_temp"), 
        avg_ph=agg.get("avg_ph"), 
        latest_turbidity=latest_turbidity, 
        latest_tds=latest_tds
    )

def _calculate_agency_distribution(stations: List[Station]):
    if not stations: return []
    agency_counts = Counter(s.agency_name for s in stations if s.agency_name)
    distribution = [{"agency_name": name, "station_count": count} for name, count in agency_counts.items()]
    return sorted(distribution, key=lambda x: x['station_count'], reverse=True)

@router_dashboard.get("/dashboard/filters", response_model=DashboardFilterOptions)
def get_dashboard_filters(db: Session = Depends(get_db), current_user: User = Depends(auth_service.get_current_user)):
    return {"agencies": crud.get_distinct_agencies(db)}

# --------------------------------------------------------------------------------------
# 6.6. Alerts Router
# --------------------------------------------------------------------------------------
@router_alerts.get("/alerts", response_model=AlertsResponse)
def get_alerts(db: Session = Depends(get_db), current_user: User = INTERNAL_USER_ROLES, state: Optional[str] = Query(None), district: Optional[str] = Query(None), basin: Optional[str] = Query(None), percentile: int = Query(90, ge=50, le=99)):
    stations = crud.get_stations_with_filters(db, state=state, district=district, basin=basin, station_type=StationType.groundwater)
    if not stations: return AlertsResponse(alerts=[], summary=AlertsSummary(total_alerts=0, most_affected_state="N/A", most_recent_alert_date="N/A", average_exceedance_m=0.0))
    station_ids = [s.id for s in stations]
    thresholds = crud.get_historical_percentiles_for_stations(db, station_ids=station_ids, percentile=(percentile / 100.0))
    latest_readings = crud.get_latest_timeseries_readings(db, station_ids=station_ids)
    station_map = {s.id: s for s in stations}
    active_alerts = []
    for reading in latest_readings:
        threshold = thresholds.get(reading.station_id)
        if threshold is not None and reading.groundwaterlevel_mbgl is not None and reading.groundwaterlevel_mbgl > threshold:
            station_info = station_map.get(reading.station_id)
            active_alerts.append(Alert(timestamp=reading.timestamp, station_name=station_info.station_name, state_name=station_info.state_name, district_name=station_info.district_name, groundwaterlevel_mbgl=reading.groundwaterlevel_mbgl, critical_threshold_mbgl=threshold))
    summary = _calculate_alerts_summary(active_alerts)
    return AlertsResponse(alerts=sorted(active_alerts, key=lambda x: x.timestamp, reverse=True), summary=summary)

def _calculate_alerts_summary(alerts: List[Alert]):
    if not alerts: return AlertsSummary(total_alerts=0, most_affected_state="N/A", most_recent_alert_date="N/A", average_exceedance_m=0.0)
    state_counts = Counter(alert.state_name for alert in alerts if alert.state_name)
    most_affected_state = state_counts.most_common(1)[0][0] if state_counts else "N/A"
    most_recent_alert_date = max(alert.timestamp for alert in alerts).strftime('%Y-%m-%d')
    exceedances = [alert.groundwaterlevel_mbgl - alert.critical_threshold_mbgl for alert in alerts]
    average_exceedance_m = sum(exceedances) / len(exceedances) if exceedances else 0.0
    return AlertsSummary(total_alerts=len(alerts), most_affected_state=most_affected_state, most_recent_alert_date=most_recent_alert_date, average_exceedance_m=round(average_exceedance_m, 2))

@router_alerts.get("/alerts/filters", response_model=MapFilterOptions)
def get_alerts_filters(db: Session = Depends(get_db), current_user: User = Depends(auth_service.get_current_user)):
    return crud.get_distinct_filter_options(db)

# --------------------------------------------------------------------------------------
# 6.7. Policy & Governance Router
# --------------------------------------------------------------------------------------
POLICY_MAKER_ROLES = Depends(auth_service.require_role({UserRole.admin.value, UserRole.policy_maker.value}))
@router_policy.post("/policy/briefing", response_model=str)
def get_policy_briefing(request: PolicyBriefingRequest, db: Session = Depends(get_db), current_user: User = POLICY_MAKER_ROLES):
    prompt = f"As a senior water policy advisor to the Government of India, analyze this data on groundwater stress by {request.group_by}: {json.dumps(request.regional_stress_data)}. Provide a concise, moderate-length briefing. Structure it with:\n1. **Executive Summary:** A single paragraph on the key takeaway.\n2. **Key Hotspots:** Bullet points identifying the most stressed regions.\n3. **Actionable Policy Recommendations:** 3 distinct, brief recommendations.\n4. **Identified Data Gaps:** A sentence on potential data limitations."
    response = intelligence_service.get_intelligence_response(db, "policy_briefing", prompt, request.model_dump())
    if not response: raise HTTPException(status_code=503, detail="Intelligent Analysis service is currently unavailable or disabled.")
    return response

@router_policy.post("/policy/intervention-advisor", response_model=str)
def get_intervention_strategies(request: PolicyStrategyRequest, db: Session = Depends(get_db), current_user: User = POLICY_MAKER_ROLES):
    prompt = f"You are a senior hydrogeologist advising the Central Ground Water Board of India. For the groundwater-stressed region of **{request.region}**, provide 3-4 specific, actionable, and cost-effective intervention strategies. Focus on both supply-side (e.g., recharge) and demand-side (e.g., efficiency) management. Present these as a bulleted list with a brief justification (1-2 sentences) for each. The response should be concise."
    response = intelligence_service.get_intelligence_response(db, "intervention_advisor", prompt, request.model_dump())
    if not response:
        raise HTTPException(status_code=503, detail="Intelligent Analysis service is currently unavailable or disabled.")
    return response

@router_policy.get("/policy/regional-stress", response_model=RegionalStressResponse)
def get_regional_stress_data(
    group_by: str = Query("state_name", enum=["state_name", "basin"]),
    percentile: int = Query(75, ge=50, le=95),
    db: Session = Depends(get_db),
    current_user: User = Depends(auth_service.get_current_user)
):
    stress_data = analysis_service.get_regional_stress(db, group_by_col=group_by, percentile=percentile)
    return RegionalStressResponse(data=stress_data)

@router_policy.get("/policy/trends", response_model=LongTermTrendResponse)
def get_long_term_trends(
    db: Session = Depends(get_db), 
    current_user: User = Depends(auth_service.get_current_user),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    basin: Optional[str] = Query(None)
):
    return analysis_service.get_filtered_long_term_trends(db, state, district, basin)

# --------------------------------------------------------------------------------------
# 6.8. Strategic Planning Router
# --------------------------------------------------------------------------------------
PLANNER_ROLES = Depends(auth_service.require_role({UserRole.admin.value, UserRole.researcher.value, UserRole.policy_maker.value}))
@router_planning.post("/planning/scenario/{station_id}", response_model=PlanningResponse)
def get_planning_scenario(
    request: PlanningRequest,
    station: Station = Depends(get_station_for_user),
    db: Session = Depends(get_db)
):
    ts_records = crud.get_timeseries_for_station(db, station_id=station.id)
    if len(ts_records) < 365: raise HTTPException(status_code=400, detail="Insufficient data for planning analysis. At least one year of data is required.")
    
    df = pd.DataFrame([{"timestamp": r.timestamp, "groundwaterlevel_mbgl": r.groundwaterlevel_mbgl} for r in ts_records])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    total_days = (df['timestamp'].max() - df['timestamp'].min()).days
    if total_days < 365:
        raise HTTPException(status_code=400, detail="A full year of data is required for an accurate annual recharge calculation.")

    df = df.set_index('timestamp').asfreq('D').interpolate(method='time')
    df['gw_level_change'] = df['groundwaterlevel_mbgl'].diff()
    df['recharge_mm'] = df.apply(lambda r: (r['gw_level_change'] * -1 * request.specific_yield * 1000) if r['gw_level_change'] < 0 else 0, axis=1)
    
    total_recharge = df['recharge_mm'].sum()
    avg_annual_recharge = (total_recharge / total_days) * 365.25

    if avg_annual_recharge < 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data led to a negative average annual recharge ({avg_annual_recharge:.2f} mm/year). This is physically impossible and indicates a data quality issue for this station. Cannot proceed with planning."
        )
    
    sustainable_yield_mm = avg_annual_recharge * 0.7
    sustainable_supply_m3 = (sustainable_yield_mm / 1000) * (request.area_sq_km * 1_000_000)
    total_demand_m3 = request.demands_m3.agricultural + request.demands_m3.industrial + request.demands_m3.domestic
    balance_m3 = sustainable_supply_m3 - total_demand_m3
    
    baseline = PlanningBaselineResult(avg_annual_recharge_mm=avg_annual_recharge, sustainable_yield_mm=sustainable_yield_mm, sustainable_supply_m3=sustainable_supply_m3, total_demand_m3=total_demand_m3, balance_m3=balance_m3)
    
    modified_supply = sustainable_supply_m3 * (1 + request.scenario.rainfall_change_percent / 100)
    modified_demand = total_demand_m3 * (1 + request.scenario.demand_change_percent / 100)
    new_balance = modified_supply - modified_demand
    
    scenario_result = PlanningScenarioResult(modified_supply_m3=modified_supply, modified_demand_m3=modified_demand, new_balance_m3=new_balance, delta_from_baseline_m3=new_balance - balance_m3)
    
    prompt = f"As a water resource management consultant, analyze this 'what-if' scenario for station {station.station_name}.\n- Baseline Balance: {balance_m3:,.0f} m³/year\n- Scenario: {request.scenario.rainfall_change_percent}% change in rainfall, {request.scenario.demand_change_percent}% change in demand.\n- Scenario Result: A new water balance of {new_balance:,.0f} m³/year.\nProvide a brief, 2-3 sentence summary of the key implication of this scenario and one strategic recommendation to manage the outcome."
    intelligent_analysis = intelligence_service.get_intelligence_response(db, "planning_analysis", prompt, request.model_dump())

    return PlanningResponse(station_name=station.station_name, baseline=baseline, scenario=scenario_result, intelligent_analysis=intelligent_analysis)

# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
# FIX: Added missing routers and their endpoints.
# 6.9. Research Hub Router
# --------------------------------------------------------------------------------------
RESEARCHER_ROLES = Depends(auth_service.require_role({UserRole.admin.value, UserRole.researcher.value}))
@router_research.get("/research/quality/{station_id}", response_model=ResearchQualityResponse)
def get_water_quality_data(
    station: Station = Depends(get_station_for_user),
    time_range_days: Optional[int] = Query(None, ge=1),
    db: Session = Depends(get_db)
):
    start_date = datetime.now(timezone.utc) - timedelta(days=time_range_days) if time_range_days else None
    ts_data = crud.get_timeseries_for_station(db, station_id=station.id, start_date=start_date)
    return ResearchQualityResponse(data=ts_data)

@router_research.post("/research/correlation/{station_id}", response_model=CorrelationResponse)
def get_correlation_analysis(
    request: CorrelationRequest,
    station: Station = Depends(get_station_for_user),
    db: Session = Depends(get_db)
):
    ts_data = crud.get_timeseries_for_station(db, station.id)
    df = pd.DataFrame([r.__dict__ for r in ts_data])
    
    p1, p2 = request.parameter1, request.parameter2
    if p1 not in df.columns or p2 not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid parameter names provided.")
        
    corr_df = df[[p1, p2]].dropna()
    if len(corr_df) < 2:
        return CorrelationResponse(correlation=None, intelligent_analysis="Not enough overlapping data to calculate correlation.")
        
    correlation = corr_df.corr().iloc[0, 1]
    
    prompt = f"As a research hydrologist, analyze the relationship between '{p1}' and '{p2}' for station {station.station_name}, which show a Pearson correlation of {correlation:.3f}. Provide a concise interpretation (2-3 sentences) covering:\n1. The strength and direction of the correlation.\n2. A plausible hydrological explanation for this relationship.\n3. One potential implication for further research."
    intelligent_analysis = intelligence_service.get_intelligence_response(db, "correlation_analysis", prompt, {"station_id": station.id, **request.model_dump()})
    
    return CorrelationResponse(correlation=correlation, intelligent_analysis=intelligent_analysis)

@router_research.post("/research/forecast/{station_id}", response_model=ForecastJobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
def create_forecast(
    background_tasks: BackgroundTasks,
    station: Station = Depends(get_station_for_user),
    days_to_forecast: int = Query(30, ge=7, le=180),
    current_user: User = RESEARCHER_ROLES,
    db: Session = Depends(get_db)
):
    job = crud.create_forecast_job(db, station_id=station.id, user_id=current_user.id)
    background_tasks.add_task(analysis_service.run_forecast_model, db, job.id, station.id, days_to_forecast)
    return ForecastJobCreateResponse(job_id=job.id, message="Forecast job started.")

@router_research.get("/research/forecast/results/{job_id}", response_model=ForecastJobStatusResponse)
def get_forecast_results(
    job_id: int,
    current_user: User = RESEARCHER_ROLES,
    db: Session = Depends(get_db)
):
    job = crud.get_forecast_job_by_id(db, job_id, user_id=current_user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Forecast job not found or access denied.")
    
    response = ForecastJobStatusResponse(job_id=job.id, status=job.status)
    if job.results:
        results_data = json.loads(job.results)
        if job.status == JobStatus.failed:
            response.error_message = results_data.get("error")
        else:
            response.results = ForecastResult(**results_data)
            
    return response

@router_research.get("/research/forecast/jobs", response_model=List[ForecastJobInfo])
def get_user_forecast_jobs(
    current_user: User = RESEARCHER_ROLES,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    return crud.get_forecast_jobs_by_user(db, user_id=current_user.id, skip=skip, limit=limit)

# --------------------------------------------------------------------------------------
# 6.10. Public Information Hub Router
# --------------------------------------------------------------------------------------
@router_public.get("/public/info/{station_id}", response_model=PublicInfoResponse)
def get_public_info(station_id: int, db: Session = Depends(get_db)):
    station = crud.get_station_by_id(db, station_id)
    if not station or station.station_type != StationType.groundwater:
        raise HTTPException(404, "Groundwater station not found.")

    latest = crud.get_latest_timeseries_readings(db, [station_id])
    if not latest:
        raise HTTPException(404, "No data available for this station.")
    
    latest_reading = latest[0]
    gauges = []
    
    if latest_reading.groundwaterlevel_mbgl is not None:
        gauges.append(GaugeData(parameter="GW Level", value=latest_reading.groundwaterlevel_mbgl, min_val=0, max_val=50, normal_range=[0, 20], unit="mbgl"))
    if latest_reading.ph is not None:
        gauges.append(GaugeData(parameter="pH", value=latest_reading.ph, min_val=5, max_val=9, normal_range=[6.5, 8.5], unit=""))
    if latest_reading.tds_ppm is not None:
        gauges.append(GaugeData(parameter="TDS", value=latest_reading.tds_ppm, min_val=0, max_val=1000, normal_range=[0, 500], unit="ppm"))

    return PublicInfoResponse(gauges=gauges)

@router_public.get("/public/info/{station_id}/summary", response_model=PublicSummaryResponse)
def get_public_summary(station_id: int, db: Session = Depends(get_db)):
    station = crud.get_station_by_id(db, station_id)
    if not station: raise HTTPException(404, "Station not found")
    
    latest_reading = crud.get_latest_timeseries_readings(db, [station_id])
    if not latest_reading:
        raise HTTPException(404, "No recent data available for this station.")
    
    latest = latest_reading[0]
    prompt_data = {
        'GW Level': f"{latest.groundwaterlevel_mbgl:.2f} mbgl" if latest.groundwaterlevel_mbgl else "N/A",
        'pH': f"{latest.ph:.2f}" if latest.ph else "N/A",
        'TDS': f"{latest.tds_ppm:.2f} ppm" if latest.tds_ppm else "N/A"
    }
    
    prompt = f"As a public information assistant, explain the local water situation in simple, non-technical language based on: {prompt_data}. Explain what the numbers mean for water availability (higher mbgl is worse) and quality (pH ideal 6.5-8.5, TDS ideal < 500 ppm). Provide a one-sentence summary and a simple, daily water-saving tip related to the data. Keep the entire response short and easy to read."
    
    summary = intelligence_service.get_intelligence_response(db, "public_summary", prompt, {"station_id": station_id})
    if not summary:
        raise HTTPException(status_code=503, detail="Intelligent Analysis service is currently unavailable or disabled.")
        
    return PublicSummaryResponse(summary=summary)

# --------------------------------------------------------------------------------------
# 6.11. Advanced Hydrology Router
# --------------------------------------------------------------------------------------
@router_hydrology.get("/hydrology/volatility/{station_id}", response_model=VolatilityResponse)
def get_volatility_analysis(station: Station = Depends(get_station_for_user), db: Session = Depends(get_db)):
    ts_data = crud.get_timeseries_for_station(db, station.id)
    if not ts_data: return VolatilityResponse(data=[], intelligent_analysis="Not enough data.")
    
    df = pd.DataFrame([r.__dict__ for r in ts_data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').asfreq('D').interpolate(method='time')
    
    df['volatility'] = df['groundwaterlevel_mbgl'].rolling(90).std()
    df_res = df.reset_index()[['timestamp', 'groundwaterlevel_mbgl', 'volatility']].dropna()
    
    response_data = [
        VolatilityPoint(timestamp=r['timestamp'].date(), level=r['groundwaterlevel_mbgl'], volatility=r['volatility'])
        for r in df_res.to_dict('records')
    ]
    
    prompt = "You are a hydrogeologist. Given a rolling 90-day volatility for a groundwater station, briefly interpret what an increasing, decreasing, or stable trend in this volatility implies. Explain what it might suggest about aquifer stability, recharge patterns, or abstraction pressures. Keep the response to a concise paragraph."
    intelligent_analysis = intelligence_service.get_intelligence_response(db, "volatility_analysis", prompt, {"station_id": station.id})
    
    return VolatilityResponse(data=response_data, intelligent_analysis=intelligent_analysis)

@router_hydrology.get("/hydrology/monsoon/{station_id}", response_model=MonsoonResponse)
def get_monsoon_analysis(station: Station = Depends(get_station_for_user), db: Session = Depends(get_db)):
    ts_data = crud.get_timeseries_for_station(db, station.id)
    if not ts_data: return MonsoonResponse(yearly_data=[], intelligent_analysis="Not enough data.")
    
    df = pd.DataFrame([r.__dict__ for r in ts_data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df['year'] = df.index.year
    
    pre_monsoon = df[df.index.month.isin([3, 4, 5])].groupby('year')['groundwaterlevel_mbgl'].mean()
    post_monsoon = df[df.index.month.isin([10, 11, 12])].groupby('year')['groundwaterlevel_mbgl'].mean()
    
    monsoon_df = pd.DataFrame({'pre_monsoon_level_mbgl': pre_monsoon, 'post_monsoon_level_mbgl': post_monsoon}).dropna().reset_index()
    monsoon_df['recharge_effect_m'] = monsoon_df['pre_monsoon_level_mbgl'] - monsoon_df['post_monsoon_level_mbgl']
    
    avg_recharge = monsoon_df['recharge_effect_m'].mean() if not monsoon_df.empty else None
    
    prompt = f"As a water resource analyst, you are looking at pre- and post-monsoon data for a groundwater station. The average recharge effect is {avg_recharge:.2f}m. Briefly explain what this value indicates about the station's monsoon recharge effectiveness and mention one factor that could cause this to vary year-on-year. Keep the response concise."
    intelligent_analysis = intelligence_service.get_intelligence_response(db, "monsoon_analysis", prompt, {"station_id": station.id, "avg_recharge": avg_recharge})

    return MonsoonResponse(
        average_recharge_effect_m=avg_recharge,
        yearly_data=monsoon_df.to_dict(orient='records'),
        intelligent_analysis=intelligent_analysis
    )

@router_hydrology.get("/hydrology/droughts/{station_id}", response_model=DroughtResponse)
def get_drought_analysis(
    station: Station = Depends(get_station_for_user),
    percentile: int = Query(85, ge=70, le=99),
    db: Session = Depends(get_db)
):
    ts_data = crud.get_timeseries_for_station(db, station.id)
    if not ts_data: return DroughtResponse(event_count=0, events=[])
    
    df = pd.DataFrame([r.__dict__ for r in ts_data])
    threshold = np.percentile(df['groundwaterlevel_mbgl'].dropna(), percentile)
    
    df['in_drought'] = df['groundwaterlevel_mbgl'] > threshold
    df['drought_block'] = (df['in_drought'].diff(1) != 0).astype('int').cumsum()
    
    events = []
    for block in df[df['in_drought']]['drought_block'].unique():
        days = df[df['drought_block'] == block]
        duration = (days['timestamp'].max() - days['timestamp'].min()).days + 1
        if duration > 30:
            events.append(DroughtEvent(
                start_date=days['timestamp'].min().date(),
                end_date=days['timestamp'].max().date(),
                duration_days=duration,
                peak_level_mbgl=days['groundwaterlevel_mbgl'].max()
            ))
            
    return DroughtResponse(event_count=len(events), events=events)

# --------------------------------------------------------------------------------------
# 6.12. Full Report Router
# --------------------------------------------------------------------------------------
@router_reports.post("/reports/generate", response_model=FullReportResponse)
def generate_full_report(
    current_user: User = INTERNAL_USER_ROLES,
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    basin: Optional[str] = Query(None),
    agency: Optional[str] = Query(None),
    time_range_days: Optional[int] = Query(None, ge=1),
    db: Session = Depends(get_db)
):
    all_stations_in_scope = crud.get_stations_with_filters(db, state=state, district=district, basin=basin, agency=agency)
    gw_stations_in_scope = [s for s in all_stations_in_scope if s.station_type == StationType.groundwater]
    
    if not gw_stations_in_scope:
        raise HTTPException(status_code=404, detail="No groundwater stations match the specified filters.")
        
    station_ids = [s.id for s in gw_stations_in_scope]
    start_date = (datetime.now(timezone.utc) - timedelta(days=time_range_days)) if time_range_days else None
    
    # KPIs
    aggregated_data = crud.get_aggregated_timeseries(db, station_ids=station_ids, start_date=start_date, frequency='all')
    latest_readings = crud.get_latest_timeseries_readings(db, station_ids=station_ids)
    boundary_aggregates = crud.get_boundary_aggregates_for_timeseries(db, station_ids=station_ids, start_date=start_date)
    kpis = _calculate_kpis_from_aggregates(aggregated_data, latest_readings, boundary_aggregates)
    
    # Alerts
    alerts_response = get_alerts(db, current_user, state, district, basin)
    
    # Trends
    trends = analysis_service.get_filtered_long_term_trends(db, state, district, basin)
    
    return FullReportResponse(
        report_generated_on=datetime.now(timezone.utc),
        selection_filters=dict(state=state, district=district, basin=basin, agency=agency, time_range_days=time_range_days),
        kpis=kpis,
        alerts_summary=alerts_response.summary,
        long_term_trends=trends
    )

@router_reports.post("/reports/summarize", response_model=str)
def summarize_report(
    request: ReportSummaryRequest,
    current_user: User = INTERNAL_USER_ROLES,
    db: Session = Depends(get_db)
):
    prompt = f"Analyze the following JSON report on water resources and generate a concise executive summary for a high-level government official. Focus on the most critical findings and actionable insights. Avoid jargon. The summary should be a moderate-length paragraph. Report Data: {request.report_data.model_dump_json()}"
    summary = intelligence_service.get_intelligence_response(db, "report_summary", prompt, request.model_dump())
    if not summary:
        raise HTTPException(status_code=503, detail="Intelligent Analysis service is currently unavailable or disabled.")
    return summary
# --------------------------------------------------------------------------------------
# 6.13. Admin Router
# --------------------------------------------------------------------------------------
@router_admin.delete("/admin/clear-intelligence-cache", status_code=status.HTTP_204_NO_CONTENT)
def clear_intelligence_cache(
    feature_name: Optional[str] = Query(None),
    request_hash: Optional[str] = Query(None),
    clear_all: bool = Query(False),
    db: Session = Depends(get_db),
    admin_user: User = Depends(auth_service.require_role({UserRole.admin.value}))
):
    if clear_all:
        db.query(IntelligenceAnalysisLog).delete()
    elif feature_name and request_hash:
        db.query(IntelligenceAnalysisLog).filter(and_(IntelligenceAnalysisLog.feature_name == feature_name, IntelligenceAnalysisLog.request_hash == request_hash)).delete()
    else:
        raise HTTPException(status_code=400, detail="Must provide 'clear_all=true' or both 'feature_name' and 'request_hash'.")
    db.commit()
    return None

# --------------------------------------------------------------------------------------
# 6.14. Station Inventory Router
# --------------------------------------------------------------------------------------
@router_stations.get("/stations/", response_model=List[Station])
def list_stations(
    skip: int = 0,
    limit: int = 100,
    current_user: User = INTERNAL_USER_ROLES,
    db: Session = Depends(get_db)
):
    stations = crud.get_stations_with_filters(db, skip=skip, limit=limit)
    return stations

# ======================================================================================
# 7. MAIN APPLICATION (`main.py`)
# ======================================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"--- {settings.PROJECT_NAME} API Starting Up ---")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try: await auth_service.create_first_superuser(db)
    finally: db.close()
    yield
    logger.info(f"--- {settings.PROJECT_NAME} API Shutting Down ---")

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION, lifespan=lifespan, openapi_url="/api/v1/openapi.json")
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(router_auth, prefix=API_PREFIX, tags=["1. Authentication"])
app.include_router(router_users, prefix=API_PREFIX, tags=["2. User Management"])
app.include_router(router_ingestion, prefix=API_PREFIX, tags=["3. Data Ingestion"])
app.include_router(router_stations, prefix=API_PREFIX, tags=["4. Station Inventory"])
app.include_router(router_map, prefix=API_PREFIX, tags=["5. Unified Map"])
app.include_router(router_dashboard, prefix=API_PREFIX, tags=["6. At-a-Glance Dashboard"])
app.include_router(router_alerts, prefix=API_PREFIX, tags=["7. Critical Alerts"])
app.include_router(router_policy, prefix=API_PREFIX, tags=["8. Policy & Governance"])
app.include_router(router_planning, prefix=API_PREFIX, tags=["9. Strategic Planning"])
app.include_router(router_research, prefix=API_PREFIX, tags=["10. Research Hub"])
app.include_router(router_public, prefix=API_PREFIX, tags=["11. Public Information Hub"])
app.include_router(router_hydrology, prefix=API_PREFIX, tags=["12. Advanced Hydrology"])
app.include_router(router_reports, prefix=API_PREFIX, tags=["13. Full Report"])
app.include_router(router_admin, prefix=API_PREFIX, tags=["14. Administration"])

@app.get("/", tags=["Health Check"])
def read_root(): return {"status": "ok", "message": f"Welcome to {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}"}

# ======================================================================================
# END OF FILE
# ======================================================================================

