# Hybrid Review Classifier

A cost-optimized review classification system that combines traditional machine learning and LLMs.

## Objective

Reduce LLM cost while maintaining high classification accuracy.

## Architecture

Review
        │
        ▼
Machine Learning Model
        │
 Confidence > Threshold ?
     ┌──────┴──────┐
     │             │
   High          Low
     │             │
     ▼             ▼
Prediction     Send to LLM
     │             │
     └──────┬──────┘
            ▼
      Final Prediction

## Tech Stack

- Python
- Scikit-learn
- Pandas
- OpenAI API
- GitHub

## Future Work

- Active Learning
- Cost Tracking
- Dashboard
