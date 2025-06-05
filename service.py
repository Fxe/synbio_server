from etl_experiment import EtlExperiment
import pandas as pd


class SynbioService:

    def __init__(self, minio, engine):
        self.minio = minio
        self.engine = engine

    def etl(self):
        # minio
        # sql engine

        plate_type = '96_shallow'  # could also be deep well plate
        start_date = '2025-04-04'
        exp_index = 1
        exp_type = 'autoALE'
        description = ''
        protocol_id = 'mock_1b_protocol'
        lab_id = 1
        contact_id = 1
        measurement_type = 'growth'

        plate_type = '96_shallow' # get_plate_type()

        etl = EtlExperiment(self.engine, self.minio)
        etl.etl(plate_type=plate_type)

    def list_experiments(self):
        pass

    def add_experiment(self):
        pass

    def query_OD(self, experiment_id, strain_id):
        '''
        Queries db for all samples from specified experiment and returns all
        associated od_measurements as pandas DataFrame. Plots all samples of
        specified strain_id.

        Args:
            experiment_id (str): Experiment id. Must be in db.
            strain_id (str): Strain id. Must be in db.
        Returns:
            DataFrame
        '''

        # DB Connection
        engine = self.engine

        # Hardcode experiment id. To be changed later
        experiment_id = 'ALE1b'

        # Check validity of passed arguments
        db_experiments = pd.read_sql(
            "SELECT experiment.id FROM experiment", engine
        )['id'].to_list()

        db_strains = pd.read_sql(
            "SELECT strain.id FROM strain", engine
        )['id'].to_list()

        if experiment_id not in db_experiments or strain_id not in db_strains:
            print(f"Check if experiment and strain are registered in the db.")

            return None

        # Hardcoded query. This can be made more flexible later.
        query = """
        SELECT 
            experiment.id,
            sample.name, sample.passage, 
            sample.strain_id, strain.long_name,
            sample.growth_condition_id, growth_condition.carbon_source,
            measurement.type,
            od_measurement.datetime, od_measurement.od, od_measurement.background
        FROM 
            experiment
            INNER JOIN sample ON sample.experiment_id = experiment.id
            INNER JOIN measurement ON measurement.sample_id = sample.name
            INNER JOIN od_measurement ON od_measurement.measurement_id = measurement.id
            INNER JOIN strain ON strain.id = sample.strain_id
            INNER JOIN growth_condition ON growth_condition.id = sample.growth_condition_id
    
        WHERE 
            (experiment.id=%(experiment)s) AND (sample.strain_id=%(strain)s)
        """

        selection = pd.read_sql(
            query, engine, params={'experiment': experiment_id, 'strain': str(strain_id)}
        ).rename(
            columns={'id': 'experiment_id',
                     'name': 'sample_name',
                     'type': 'measurement_type',
                     'long_name': 'strain_name'}
        )

        return selection
