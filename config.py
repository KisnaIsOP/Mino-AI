class Config:
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'your-secret-key'  # Change this in production

class DevelopmentConfig(Config):
    DEBUG = True
    DEVELOPMENT = True
    TEMPLATES_AUTO_RELOAD = True
    
class ProductionConfig(Config):
    DEBUG = False
    
class TestingConfig(Config):
    TESTING = True
