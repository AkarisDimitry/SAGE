try:
    from sage_lib.partition.PartitionManager import PartitionManager
except ImportError as e:
    import sys
    sys.stderr.write(f"An error occurred while importing PartitionManager: {str(e)}\n")
    del sys

class PositionEditor_builder(PartitionManager):
    def __init__(self, file_location:str=None, name:str=None, **kwargs):
        super().__init__(name=name, file_location=file_location)
        
    def handleVacancy(self, container, values, container_index, file_location=None):
        sub_directories, containers = [], []

        container_copy = self.copy_and_update_container(container, '/Vacancy', file_location)
        container_copy.AtomPositionManager = CrystalDefectGenerator(Periodic_Object=container_copy.AtomPositionManager)
        all_vacancy_configs, all_vacancy_label = container_copy.AtomPositionManager.generate_all_vacancies()

        for cv_i, (vacancy_configs, vacancy_label) in enumerate(zip(all_vacancy_configs, all_vacancy_label)):
            container_copy2 = copy.deepcopy(container_copy)
            container_copy2.AtomPositionManager = vacancy_configs
            container_copy2.file_location = f'{container_copy.file_location}/{cv_i}_{vacancy_label}'
            sub_directories.append(f'{cv_i}_{vacancy_label}')
            containers.append(container_copy2)
        
        self.generate_execution_script_for_each_container(sub_directories, container.file_location + '/Vacancy')
        return containers

    def handleRattle(self, container, values, container_index, file_location=None):
        sub_directories, containers = [], []

        for v in values: 
            for n in range(v['N']):
                for std in v['std']:
                    container_copy = self.copy_and_update_container(container, f'/rattle/{std}/{n}', file_location)
                    container_copy.AtomPositionManager.rattle(stdev=std, seed=n)
                    
                    sub_directories.append(f'{std}/{n}')
                    containers.append(container_copy)
        
        self.generate_execution_script_for_each_container(sub_directories, container.file_location + '/rattle')
        return containers


    def handleCompress(self, container, values, container_index, file_location=None):
        sub_directories, containers = [], []

        compress_vector = self.interpolate_vectors(values['compress_min'], values['compress_max'], values['N'])

        for v_i, v in enumerate(compress_vector): 
            container_copy = self.copy_and_update_container(container, f'/compress/{v_i}', file_location)
            container_copy.AtomPositionManager.compress(compress_factor=v, verbose=False)
                
            sub_directories.append(f'/{v_i}')
            containers.append(container_copy)
    
        self.generate_execution_script_for_each_container(sub_directories, container.file_location + '/compress')
        return containers

