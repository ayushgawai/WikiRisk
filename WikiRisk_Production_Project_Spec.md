One Paragraph Abstract: Problem Definition
Every day Wikipedia receives millions of edits which include both high-quality content and
vandalism plus misinformation and low-value changes. The automatic detection of potentially dangerous edits would provide real-time benefits to both moderators and downstream
users who consume content. The project aims to create a complete big data solution which
uses distributed SparkML classification model training to assess ”edit risk” the probability
that an edit will be viewed as suspicious or will face reversion through analysis of Wikipedia
dump data that exceeds 10GB after decompression. The system will execute historical
dump data processing through batch mode to create features and develop supervised machine learning models while using MLflow to manage experiment tracking and model version
control and finally it will use Spark Structured Streaming to deploy the complete model
which connects to Wikimedia’s real-time EventStreams API. A FastAPI backend will deliver
predictions which users can view through a simple interface that enables them to track and
filter dangerous edits in real time. The project shows how to handle large-scale data processing through distributed systems while demonstrating streaming data analysis and creating
repeatable machine learning processes and implementing a complete operational big data
machine learning system.
Dataset Links and Short Description
1. English Wikipedia Dumps (Large-Scale Historical Data)
https://dumps.wikimedia.org/enwiki/
Description: The English Wikipedia dump contains all historical revision records for
every article with complete editor identification, timestamp information, revision size
details, editing comments, and complete revision content information. The dumps are
available to the public in compressed XML format, which requires multiple systems to
extract data elements from the files. We will select sufficient dump files so that the
decompressed dataset exceeds 10GB, ensuring genuine big data processing with Spark.
The dataset will serve as the foundation for developing extensive features and training
supervised machine learning models.
2. Wikimedia EventStreams – RecentChange (Live Streaming Data)
https://stream.wikimedia.org/v2/stream/recentchange
Description: Wikimedia EventStreams provides a real-time stream of edit events across
Wikipedia projects. The event contains metadata which includes user information and
edit size delta and revision ID and comment. The streaming source will enable realtime inference through Spark Structured Streaming because it enables live prediction
of edit risk.


# DATA 228 (SP26) -- Production-Grade End-to-End Big Data Project Specification

## Project Name

WikiRisk: Real-Time Wikipedia Edit Risk & Knowledge Integrity Monitoring
System

------------------------------------------------------------------------

# 1. STRICT COMPLIANCE WITH COURSE REQUIREMENTS

This project MUST satisfy ALL of the following (non-optional):

## Dataset Requirements

-   Publicly available dataset

-   ≥ 10GB after decompression (excluding irrelevant binaries)

-   Requires distributed processing (Spark)

-   Tens of thousands of records

## Technical Requirements

-   Data Collection & Processing (Batch AND Streaming)
-   Machine Learning Model Building
-   Model Deployment (FastAPI)
-   Frontend / Lightweight UI
-   Reproducibility (MLflow pipeline + experiment tracking)
-   Real collaborative Git repository with PRs, issues, reviews
-   Final report in engineering research paper format
-   Live demo during presentation

## Extra Points (Must Implement)

-   Production-grade architecture
-   Containerization (Docker)
-   Structured logging & monitoring
-   AI-enhanced explainability (OpenAI integration)
-   Clean minimalist UI like production SaaS dashboard

------------------------------------------------------------------------

# 2. PROJECT OVERVIEW

Wikipedia is open-edit and highly trusted. However, edits occur at
massive scale and include vandalism, misinformation, coordinated
manipulation, and destructive changes.

We will build:

-   A distributed batch ML pipeline using Spark on ≥10GB Wikipedia dumps
-   A real-time streaming inference system using Spark Structured
    Streaming
-   A deployed FastAPI backend
-   A production-style web dashboard
-   MLflow-based reproducibility
-   AI-powered explanation layer (OpenAI)

------------------------------------------------------------------------

# 3. DATA SOURCES

## Batch Dataset (≥10GB after decompression)

Source: https://dumps.wikimedia.org/enwiki/

Use: - pages-meta-history or pages-articles dumps - Download sufficient
partitions to exceed 10GB uncompressed

Maintain: - dataset_manifest.json with: - URLs - compressed size -
decompressed size - number of records

## Streaming Dataset

Source: https://stream.wikimedia.org/v2/stream/recentchange

Real-time JSON event stream of live edits.

------------------------------------------------------------------------

# 4. ARCHITECTURE

Batch Training Layer Wikipedia Dumps → Spark Batch → Feature Engineering
→ SparkML Training → MLflow Tracking → Model Registry

Streaming Layer RecentChange Stream → Spark Structured Streaming →
Feature Extraction → Load MLflow Model → Real-Time Risk Scoring →
Delta/Parquet Sink

Serving Layer FastAPI → REST Endpoints → UI Dashboard

AI Layer OpenAI API → Natural-language explanation of why edit is risky

------------------------------------------------------------------------

# 5. TECH STACK

Core: - Apache Spark (Spark SQL, Spark MLlib) - Spark Structured
Streaming - MLflow

Backend: - FastAPI - Pydantic - Async endpoints

Frontend: - Streamlit - Clean white/light theme

AI: - OpenAI API

DevOps: - Docker - docker-compose - Structured logging

Storage: - Parquet / Delta

------------------------------------------------------------------------

# 6. PRODUCTION FOLDER STRUCTURE

wikirisk/ │ ├── docker-compose.yml ├── Makefile ├── requirements.txt ├──
README.md │ ├── src/ │ ├── config/ │ ├── batch/ │ ├── streaming/ │ ├──
ml/ │ ├── serving/ │ ├── ui/ │ ├── ai/ │ └── common/ │ ├── manifests/ │
├── dataset_manifest.json │ └── model_manifest.json │ ├── tests/ │ └──
docs/

------------------------------------------------------------------------

# 7. IMPLEMENTATION PLAN

## Phase 1 -- Infrastructure Setup

-   Repo structure
-   Docker + MLflow setup
-   Environment config
-   Logging
-   README

## Phase 2 -- Batch Pipeline

-   Download dumps
-   Confirm ≥10GB decompressed
-   Spark parsing
-   Feature engineering
-   Weak label creation
-   Logistic Regression training
-   MLflow logging

## Phase 3 -- Streaming

-   Connect to EventStreams
-   Feature alignment
-   Load MLflow model
-   Real-time scoring
-   Checkpointing

## Phase 4 -- Backend API

Endpoints: - /health - /recent - /edit/{id} - /explain/{id}

## Phase 5 -- Frontend

-   Live table
-   Risk filtering
-   Detail view
-   AI explanation

## Phase 6 -- AI Layer

-   Send context to OpenAI
-   Cache explanation
-   Store with edit ID

------------------------------------------------------------------------

# 8. MACHINE LEARNING DETAILS

Model: - SparkML Logistic Regression

Features: - TF-IDF on edit_comment - TF-IDF on page_title -
length_delta - is_anon - namespace - hour_of_day

Metrics: - AUC - PR-AUC - Confusion matrix

------------------------------------------------------------------------

# 9. REPRODUCIBILITY

-   MLflow tracking
-   Model registry
-   Versioned dataset manifest
-   Re-train command
-   Reproduce via run_id

------------------------------------------------------------------------

# 10. REPORT REQUIREMENTS

Must include: - Abstract - Architecture - Dataset description - ML
methodology - Results - Discussion - Conclusion - References

------------------------------------------------------------------------

# 11. COLLABORATION

-   GitHub repo
-   PR workflow
-   Issues tracking
-   Contribution slide

------------------------------------------------------------------------

# 12. ACCEPTANCE CHECKLIST

Complete only if:

-   Dataset ≥10GB confirmed
-   Spark batch runs
-   MLflow logs
-   Streaming works
-   FastAPI serves predictions
-   UI functional
-   AI explanations working
-   Dockerized
-   Report written
-   Demo successful

------------------------------------------------------------------------

# FINAL INSTRUCTION

Build step-by-step. Test each phase. Production-quality only. Clean
modular code. Full documentation.



professors: strict requirement: 

DATA 228 Group Project Guidelines (SP26)

Description

Using the big data technologies you are learning in this class (e.g. Spark, etc.), obtain a publicly available large dataset and derive new and non-trivial insights by performing an end-to-end big data analysis (batch and/or streaming)
You have a lot of freedom to explore and pursue a project that you feel excited about, as long as you follow the guidelines stated below. Use the freedom to come up with a cool idea.

Deliverables
Project proposal
Project live presentation
Project code
Project final report

Guidelines
The size of the dataset should be meaningfully large (e.g. > ~ 10 GBs uncompressed, and having > tens of thousands of records)
The 10 GB is not a hard rule; obtain a meaningfully large dataset
Binary data that’s not relevant to your project doesn’t count; e.g. images, videos, etc.
Your project must include:
Data Collection & Processing (Batch and/or Streaming)
Machine Learning Model Building 
Model Deployment (FastAPI for example)
Frontend / Lightweight UI
Reproducibility (Using Pipeline technology such as MLflow)
The analysis must not be identical or similar to other publicly available analyses that
have been previously performed on the same dataset
The proposal can be a short paragraph of the summary of the project you will pursue
It doesn’t need to be very detailed; a high-level description is sufficient
If you need to change the project direction during the semester, let me know
The final presentation must involve a demo session that supports the conclusion of the project
It does NOT need to cover the entirety of your work; it can be a part that illustrates the value of your work or the insight
The demo does not need to run on a real cluster; a local operation is fine
The presentation should describe the project succinctly and it should present the
conclusion in a convincing manner
You must use a real source control repository for people to collaborate on the code as well as any project documents (e.g. GitHub repository)
You should demonstrate a real collaborative team effort; for example, commits, PRs (pull requests), issues, reviews, discussions, and merges
There should be a sustained period during which the collaboration occurs on the source control repository
Your final report should follow a typical (engineering) research paper style and be thorough in capturing all your work and conclusions
The report should have sections such as the abstract, the main body of your project, the conclusion, and the references (if any)
You must work together as a team, and each member must make meaningful and similarly-sized contributions to the project
You are encouraged to add a slide that lists contributions from all the tea members
You are welcome to add and use other technologies that are not part of this class, such as AI, visualization, other data technology products, etc.
The extra ingredients are not central to the overall performance; however, it may
be considered part of the overall effort and rigor of the project and result in small
extra points for the group

