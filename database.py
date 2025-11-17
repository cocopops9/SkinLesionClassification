"""
Database models and utilities for user authentication and image management.
Implements SQLAlchemy ORM for persistent storage with strict data validation.
"""

import os
import uuid
import bcrypt
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.exc import IntegrityError
import json

# Database configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'melanoma_app.db')}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, connect_args={'check_same_thread': False})
# expire_on_commit=False prevents DetachedInstanceError when accessing objects after session close
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


class User(Base):
    """User model with password hashing using bcrypt for security."""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationship to images
    images = relationship("ImageRecord", back_populates="user", cascade="all, delete-orphan")
    
    def set_password(self, password: str) -> None:
        """Hash password using bcrypt with automatic salt generation."""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash."""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))


class ImageRecord(Base):
    """Image record model linking images to users with full analysis metadata."""
    __tablename__ = "images"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow)
    
    # Prediction results
    diagnosis = Column(String(100), nullable=True)
    lesion_type = Column(String(50), nullable=True)
    confidence_scores = Column(Text, nullable=True)  # JSON string
    model_used = Column(String(100), nullable=True)
    
    # Explainability data
    explanation_text = Column(Text, nullable=True)
    feature_importance = Column(Text, nullable=True)  # JSON string
    heatmap_path = Column(String(500), nullable=True)
    
    # Image validation
    is_valid_skin_image = Column(Boolean, default=True)
    validation_message = Column(String(500), nullable=True)
    
    # Processing metadata
    processing_time = Column(Float, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="images")
    
    def set_confidence_scores(self, scores: Dict[str, float]) -> None:
        """Store confidence scores as JSON."""
        self.confidence_scores = json.dumps(scores)
    
    def get_confidence_scores(self) -> Dict[str, float]:
        """Retrieve confidence scores from JSON."""
        return json.loads(self.confidence_scores) if self.confidence_scores else {}
    
    def set_feature_importance(self, features: Dict[str, Any]) -> None:
        """Store feature importance data as JSON."""
        self.feature_importance = json.dumps(features)
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Retrieve feature importance data from JSON."""
        return json.loads(self.feature_importance) if self.feature_importance else {}


class DatabaseManager:
    """Database management utilities with connection pooling and error handling."""
    
    @staticmethod
    def init_db():
        """Initialize database tables."""
        Base.metadata.create_all(bind=engine)
    
    @staticmethod
    def get_session() -> Session:
        """Get database session with proper cleanup."""
        return SessionLocal()
    
    @staticmethod
    def create_user(username: str, password: str, email: str) -> Optional[User]:
        """Create new user with validation."""
        session = SessionLocal()
        try:
            # Validate input
            if not username or len(username) < 3:
                raise ValueError("Username must be at least 3 characters")
            if not email or '@' not in email:
                raise ValueError("Invalid email address")

            user = User(username=username, email=email)
            user.set_password(password)
            session.add(user)
            session.commit()
            session.refresh(user)
            # Expunge to detach from session before closing
            session.expunge(user)
            return user
        except IntegrityError:
            session.rollback()
            return None
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[User]:
        """Authenticate user and update last login."""
        session = SessionLocal()
        try:
            user = session.query(User).filter_by(username=username, is_active=True).first()
            if user and user.verify_password(password):
                user.last_login = datetime.utcnow()
                session.commit()
                # Expunge to detach from session before closing
                session.expunge(user)
                return user
            return None
        finally:
            session.close()
    
    @staticmethod
    def save_image_record(user_id: str, image_data: Dict[str, Any]) -> ImageRecord:
        """Save image analysis record with full metadata."""
        session = SessionLocal()
        try:
            record = ImageRecord(
                user_id=user_id,
                **image_data
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            # Expunge to detach from session before closing
            session.expunge(record)
            return record
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    @staticmethod
    def get_user_images(user_id: str, limit: int = 100) -> List[ImageRecord]:
        """Retrieve user's image history."""
        session = SessionLocal()
        try:
            records = session.query(ImageRecord).filter_by(
                user_id=user_id
            ).order_by(ImageRecord.upload_time.desc()).limit(limit).all()
            # Expunge all records to detach from session before closing
            for record in records:
                session.expunge(record)
            return records
        finally:
            session.close()
    
    @staticmethod
    def get_image_by_id(image_id: str, user_id: str) -> Optional[ImageRecord]:
        """Retrieve specific image record with user validation."""
        session = SessionLocal()
        try:
            record = session.query(ImageRecord).filter_by(
                id=image_id,
                user_id=user_id
            ).first()
            if record:
                # Expunge to detach from session before closing
                session.expunge(record)
            return record
        finally:
            session.close()


# Initialize database on module import
DatabaseManager.init_db()
