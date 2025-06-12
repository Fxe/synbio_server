from etl_experiment import EtlExperiment, AddPlateParameters
import pandas as pd
import io
from minio.error import S3Error


class SynbioService:

    def __init__(self, minio, minio_bucket, engine):
        self.minio = minio
        self.engine = engine
        self.minio_bucket = minio_bucket

    def add_plate_data(self, exp_id, exp_index, exp_type, filename, data_well, start_date,
                       lab_id, contact_id,
                       auto_fill=True, description='', operation_id=None,
                       plate_id=None, plate_index=None, plate_transfer=None, plate_timestamp=None):

        print(exp_id, exp_index, exp_type, filename, data_well, start_date)
        print(lab_id, contact_id, auto_fill, description)

        bucket_name = self.minio_bucket
        object_name = f'/exp/{exp_id}/plate/{filename}'

        print('bucket', bucket_name, 'object', object_name)

        try:
            stat = self.minio.stat_object(bucket_name, object_name)
            print(stat)
            #raise FileExistsError(f"Object '{object_name}' already exists in bucket '{bucket_name}'.")

        except S3Error as ex_s3:
            print(ex_s3)

        self.minio.put_object(bucket_name, object_name,
                              io.BytesIO(data_well),
                              len(data_well),
                              content_type="text/plain"
                              )
        print(f"'{object_name}' uploaded successfully to bucket '{bucket_name}'.")

        if auto_fill:
            operation_id = f"{exp_id}_operation"
            import re
            fname_pattern = re.compile(
                r'(?P<experiment>\w+)_(?P<timestamp>\d+)_(?P<uid>\w+)_(?P<plate>\d+)_(?P<transfer>[1-3])_(?P<timepoint>\d+).txt'
            )
            match = fname_pattern.match(filename)
            plate_id = str(match.group('uid'))
            plate_index = int(match.group('plate'))
            plate_transfer = int(match.group('transfer'))
            plate_timestamp = int(match.group('timestamp'))
        else:
            if operation_id is None:
                raise ValueError(f'operation ID: {operation_id}')
            if plate_id is None or plate_index is None or plate_index is None or plate_timestamp is None:
                raise ValueError(f'Missing plate info')

        plate_parameters = AddPlateParameters('mock_1b_protocol',
                                              operation_id,
                                              '96_shallow',
                                              'plate_layouts/',
                                              'growth', lab_id, contact_id,
                                              plate_id=plate_id, plate_index=plate_index,
                                              plate_transfer=plate_transfer, plate_timestamp=plate_timestamp,
                                              client=self.engine)

        # Upload this experiment and its operation (i.e., procedure) to the db.

        self.add_operation(plate_parameters.operation_id, plate_parameters.protocol_id,
                           plate_parameters.lab_id, plate_parameters.contact_id,
                           start_date,
                           fail_on_exist=False)

        self.add_experiment(exp_id, plate_parameters.operation_id, exp_index, exp_type,
                            start_date, description=description,
                            fail_on_exist=False)

        response = None
        try:
            response = self.minio.get_object(self.minio_bucket, object_name)
            df_plate_data = pd.read_csv(io.BytesIO(response.data), header=None)
            print(df_plate_data)
            etl = EtlExperiment(self.engine, self.minio)
            etl.etl_plate(df_plate_data, plate_parameters, start_date, exp_id, exp_index)
        except Exception as e:
            raise Exception(e)
        finally:
            response.close()
            response.release_conn()

    def lab_list(self):
        return pd.read_sql("SELECT * FROM lab", self.engine)

    def lab_get(self):
        pass

    def lab_add(self):
        pass

    def lab_remove(self):
        pass

    def people_list(self):
        return pd.read_sql("SELECT * FROM people", self.engine)

    def people_get(self):
        pass

    def people_add(self):
        pass

    def people_remove(self):
        pass

    def add_operation(self, operation_id, protocol_id, lab_id, contact_id, start_date,
                      fail_on_exist=True):
        operation_dict = {
            'id': [operation_id],
            'protocol_id': [protocol_id],
            'lab_id': [lab_id],
            'contact_id': [contact_id],
            'timestamp': [start_date]
        }

        exp_operation_df = pd.DataFrame.from_dict(operation_dict)
        ct = pd.read_sql(f"SELECT * FROM operation WHERE id = '{list(exp_operation_df['id'])[0]}'",
                         self.engine).shape[0]

        if ct > 0 and fail_on_exist:
            raise ValueError('Duplicate Operation')
        if ct == 0:
            exp_operation_df.to_sql('operation', self.engine, index=False, if_exists='append')

    def add_experiment(self, experiment_id, operation_id, exp_index, exp_type, start_date,
                       description='', fail_on_exist=True):
        exp_dict = {
            'id': [experiment_id],
            'type': [exp_type],
            'start_date': [start_date],
            'index': [exp_index],
            'description': [description],
            'operation_id': [operation_id]
        }

        new_exp_df = pd.DataFrame.from_dict(exp_dict)
        ct = pd.read_sql(f"SELECT * FROM experiment WHERE id = '{list(new_exp_df['id'])[0]}'",
                         self.engine).shape[0]
        if ct > 0 and fail_on_exist:
            raise ValueError('Duplicate Operation')
        if ct == 0:
            new_exp_df.to_sql('experiment', self.engine, index=False, if_exists='append')

    def query_od(self, experiment_id, strain_id):
        """
        Queries db for all samples from specified experiment and returns all
        associated od_measurements as pandas DataFrame. Plots all samples of
        specified strain_id.

        Args:
            experiment_id (str): Experiment id. Must be in db.
            strain_id (str): Strain id. Must be in db.
        Returns:
            DataFrame
        """

        # Check validity of passed arguments
        db_experiments = pd.read_sql(
            "SELECT experiment.id FROM experiment", self.engine
        )['id'].to_list()

        db_strains = pd.read_sql(
            "SELECT strain.id FROM strain", self.engine
        )['id'].to_list()

        print('!')
        print(db_experiments)
        print(db_strains)

        if experiment_id not in db_experiments or int(strain_id) not in db_strains:
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
            query, self.engine, params={'experiment': experiment_id, 'strain': str(strain_id)}
        ).rename(
            columns={'id': 'experiment_id',
                     'name': 'sample_name',
                     'type': 'measurement_type',
                     'long_name': 'strain_name'}
        )

        return selection

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
