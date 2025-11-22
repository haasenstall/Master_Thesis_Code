# model/database_models.py

from sqlalchemy import Column, Integer, Text, Boolean, Date, ForeignKey, String, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Platform(Base):
    __tablename__ = 'platforms'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    url = Column(Text)
    description = Column(Text)
    
    # Relationships
    requests = relationship("Request", back_populates="platform")
    data_access = relationship("DataAccess", back_populates="platform")
    public_disclosures = relationship("PublicDisclosure", back_populates="platform")  # Add this

class ClinicalTrial(Base):
    __tablename__ = 'clinical_trials'
    
    nct_id = Column(Text, primary_key=True)
    title = Column(Text)
    phase = Column(Text)
    status = Column(Text)
    enrollment = Column(Integer)
    study_type = Column(Text)
    start_date = Column(Date)
    completion_date = Column(Date)
    date_published = Column(Text)
    plan_to_share = Column(Text)
    
    # Relationships
    eligibility_criteria = relationship("EligibilityCriteria", back_populates="clinical_trial")
    outcomes = relationship("Outcome", back_populates="clinical_trial")
    design = relationship("Design", back_populates="clinical_trial")
    data_access = relationship("DataAccess", back_populates="clinical_trial")
    request_links = relationship("RequestTrialLink", back_populates="clinical_trial")
    countries = relationship("Country", back_populates="clinical_trial")
    conditions = relationship("Condition", back_populates="clinical_trial")
    intervention_mesh_terms = relationship("Intervention_Mesh_Term", back_populates="clinical_trial")
    interventions = relationship("Intervention", back_populates="clinical_trial")
    sponsors = relationship("Sponsor", back_populates="clinical_trial")
    institution = relationship("Institution", back_populates="clinical_trial")
    investigators = relationship("Investigator", back_populates="clinical_trial")
    documents = relationship("Document", back_populates="clinical_trial")  # Add this line

class EligibilityCriteria(Base):
    __tablename__ = 'eligibility_criteria'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id'), nullable=False)
    sampling_method = Column(Text)
    gender = Column(Text)
    minimum_age = Column(Text)
    maximum_age = Column(Text)
    healthy_volunteers = Column(Text)
    
    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="eligibility_criteria")

class Institution(Base):
    __tablename__ = 'institution'
    
    id = Column(Integer, primary_key=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'))
    name = Column(Text)
    city = Column(Text)
    country = Column(Text)
    
    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="institution")

class Investigator(Base):
    __tablename__ = 'investigator'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'))
    name = Column(Text)
    role = Column(Text)
    affiliation = Column(Text, nullable=False)
    
    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="investigators")

class Outcome(Base):
    __tablename__ = 'outcomes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id'), nullable=False)
    outcome_type = Column(Text, nullable=False)  # 'primary' or 'secondary'
    
    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="outcomes")

class Design(Base):
    __tablename__ = 'design'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id'), nullable=False)
    allocation = Column(Text)
    intervention_model = Column(Text)
    observational_model = Column(Text)
    primary_purpose = Column(Text)
    
    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="design")

class Country(Base):
    __tablename__ = 'countries'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'), nullable=False)
    name = Column(Text, nullable=False)

    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="countries")

class Condition(Base):
    __tablename__ = 'conditions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'), nullable=False)
    condition = Column(Text, nullable=False)

    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="conditions")

class Intervention_Mesh_Term(Base):
    __tablename__ = 'interventions_mesh_terms'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'), nullable=False)
    intervention = Column(Text, nullable=False)

    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="intervention_mesh_terms")

class Intervention(Base):
    __tablename__ = 'interventions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'), nullable=False)
    intervention_type = Column(Text)
    name = Column(Text, nullable=False)

    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="interventions")

class Sponsor(Base):
    __tablename__ = 'sponsors'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'), nullable=False)
    name = Column(Text, nullable=False)
    agency_class = Column(Text)
    type = Column(Text)

    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="sponsors")

class DataAccess(Base):
    __tablename__ = 'data_access'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trial_nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'))
    platform_id = Column(Integer, ForeignKey('platforms.id', ondelete='CASCADE'))
    platform_name = Column(Text)
    
    # Relationships
    clinical_trial = relationship("ClinicalTrial", back_populates="data_access")
    platform = relationship("Platform", back_populates="data_access")

class Request(Base):
    __tablename__ = 'requests'
    
    request_id = Column(Text, primary_key=True)
    platform_id = Column(Integer, ForeignKey('platforms.id', ondelete='CASCADE'))
    title = Column(Text)
    investigator = Column(Text)
    institution = Column(Text)
    date_of_request = Column(Date)
    number_of_trials_requested = Column(Text)

    
    # Relationships
    platform = relationship("Platform", back_populates="requests")
    trial_links = relationship("RequestTrialLink", back_populates="request")
    public_disclosures = relationship("PublicDisclosure", back_populates="request")  # Add this

class RequestTrialLink(Base):
    __tablename__ = 'request_trial_links'
    
    request_id = Column(Text, ForeignKey('requests.request_id', ondelete='CASCADE'), primary_key=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'), primary_key=True)
    
    # Relationships
    request = relationship("Request", back_populates="trial_links")
    clinical_trial = relationship("ClinicalTrial", back_populates="request_links")

class PublicDisclosure(Base):
    __tablename__ = 'public_disclosures'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    platform_id = Column(Integer, ForeignKey('platforms.id', ondelete='CASCADE'))
    request_id = Column(Text, ForeignKey('requests.request_id', ondelete='CASCADE'), nullable=False)
    title = Column(Text, nullable=False)
    doi = Column(Text, unique=True)
    publication_date = Column(Date)
    authors = Column(Text)
    instituition = Column(Text)

    # Relationships
    request = relationship("Request", back_populates="public_disclosures")
    platform = relationship("Platform", back_populates="public_disclosures")
    focal_references = relationship("FocalReference", back_populates="publication")
    future_citations = relationship("FutureCitation", back_populates="publication")

class FocalReference(Base):
    __tablename__ = 'focal_references'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    reference_doi = Column(Text, nullable=False)
    origin_doi = Column(Text, ForeignKey('public_disclosures.doi', ondelete='CASCADE'), nullable=False)
    
    # Relationships
    publication = relationship("PublicDisclosure", back_populates="focal_references")

class FutureCitation(Base):
    __tablename__ = 'future_citations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    citation_doi = Column(Text, nullable=False)
    origin_doi = Column(Text, ForeignKey('public_disclosures.doi', ondelete='CASCADE'), nullable=False)
    
    # Relationships
    publication = relationship("PublicDisclosure", back_populates="future_citations")

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(Text, ForeignKey('clinical_trials.nct_id', ondelete='CASCADE'), nullable=False)
    document_type = Column(Text)

    # Relationships
    clinical_trial = relationship("ClinicalTrial")

class MeshTerm(Base):
    __tablename__ = 'mesh_terms'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    qualifier = Column(Text)
    tree_number = Column(Text, nullable=False)
    mesh_term = Column(Text, nullable=False)
    downcase_mesh_term = Column(Text, nullable=False)

