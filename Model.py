class Model:
   __instance = None
   @staticmethod
   def getInstance():
      if Model.__instance == None:
         Model()
      return Model.__instance

   def __init__(self):
      if Model.__instance != None:
         raise Exception("This class is a singleton!")
      else:
         Model.__instance = self
      self.set_name('ssd_inception_v2_coco_2018_01_28')
      self.set_bool_custom_trained(False)

   def get_name(self):
       return self.__name
   def set_name(self, name):
       self.__name = name
   def get_bool_custom_trained(self):
       return self.__bool_custom_trained
   def set_bool_custom_trained(self, bool_custom_trained):
       self.__bool_custom_trained = bool_custom_trained

