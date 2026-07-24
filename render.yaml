services:
  - type: web
    name: football-analytics-platform
    env: docker
    plan: free
    dockerfilePath: ./Dockerfile
    envVars:
      - key: PORT
        value: 8501
      # Add these once you wire up live data / billing (all optional):
      # - key: FOOTBALL_API_KEY
      #   sync: false
      # - key: ODDS_API_KEY
      #   sync: false
      # - key: STRIPE_SECRET_KEY
      #   sync: false
