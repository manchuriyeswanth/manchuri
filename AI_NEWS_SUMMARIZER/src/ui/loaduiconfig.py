from configparser import ConfigParser

class Config:
    def __init__(self, config_file = "src/ui/uiconfig.ini"):
        self.config = ConfigParser()
        self.config.read(config_file)
    
    def get_page_title(self):
        return self.config["DEFAULT"].get("PAGE_TITLE")
    
    def get_llm_options(self):
        return self.config["DEFAULT"].get("LLM_OPTIONS").split(",")
    
    def get_use_case(self):
        return self.config["DEFAULT"].get("USE_CASE")
    
    def get_llm_model(self):
        return self.config["DEFAULT"].get("MODEL_NAME").split(",")