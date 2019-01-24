# import standard packages
import sys
import os
from datetime import datetime, timedelta

# import airflow module
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# RMQ: do not correct sys.path to get those imports working <MV 2018-09-06>
# the problem with adding '../src' to sys.path is that this limit deployment options.
# we shall not manage those deployments details from quality code.
# in case you are having problems with this, consider adding a .pth file
# inside deployment environment.

# import code base modules
import dag_helpers
import tasks
import training_tasks

# load training_config to get training parameter setting
training_config = dag_helpers.get_training_config()

# initialize default arguments
default_args = {
    'owner': 'haaluser',
    'depends_on_past': False,
    'start_date': datetime(2018, 2, 13),
    'email': ['fissette.julien@bcg.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}
# initialize training_dag
training_dag = DAG(
    os.path.splitext(os.path.basename(__file__))[0],
    description='Training model workflow',
    default_args=default_args,
    schedule_interval=None,
)

with training_dag as dag:
    get_global_model_inputs = PythonOperator(
        task_id='get_global_model_inputs',
        python_callable=training_tasks.get_global_model_inputs,
        op_kwargs=training_config,
        provide_context=True
    )

    get_product_table = PythonOperator(
        task_id='get_product_table',
        python_callable=tasks.get_product_table,
        op_kwargs=training_config,
        provide_context=True
    )

    get_training_stock_master = PythonOperator(
        task_id='get_training_stock_master',
        python_callable=training_tasks.get_training_stock_master,
        op_args=[training_config],
        op_kwargs=training_config,
        provide_context=True
    )

    get_training_sales_master = PythonOperator(
        task_id='get_training_sales_master',
        python_callable=training_tasks.get_training_sales_master,
        op_args=[training_config],
        op_kwargs=training_config,
        provide_context=True
    )

    get_global_model_inputs >> get_product_table >> [get_training_sales_master, get_training_stock_master]

    get_training_df_reweighing_distribution = PythonOperator(
        task_id='get_training_df_reweighing_distribution',
        python_callable=training_tasks.get_training_df_reweighing_distribution,
        op_args=[training_config],
        op_kwargs=training_config,
        provide_context=True
    )


    # For each pm to train a specific model for
    for training_pm in training_config['pms_to_train_for']:

        get_training_set = PythonOperator(
            task_id=training_pm['training_pm'] + '_get_training_set',
            python_callable=training_tasks.get_training_set,
            op_args=[training_pm],
            op_kwargs=training_config,
            provide_context=True
        )

        prep_training_df = PythonOperator(
            task_id=training_pm['training_pm'] + '_prep_training_df',
            python_callable=training_tasks.prep_training_df,
            op_args=[training_pm],
            op_kwargs=training_config,
            provide_context=True
        )

        train_model = PythonOperator(
            task_id=training_pm['training_pm'] + '_train_model',
            python_callable=training_tasks.train_model,
            op_args=[training_config, training_pm],
            op_kwargs=training_config,
            provide_context=True
        )

        store_model = PythonOperator(
            task_id=training_pm['training_pm'] + '_store_model',
            python_callable=training_tasks.store_model,
            op_args=[training_pm],
            op_kwargs=training_config,
            provide_context=True
        )

        write_training_config_info = PythonOperator(
            task_id=training_pm['training_pm'] + '_write_training_config_info',
            python_callable=training_tasks.write_training_config_info,
            op_args=[training_pm],
            op_kwargs=training_config,
            provide_context=True
        )

        [get_training_sales_master, get_training_stock_master, get_training_df_reweighing_distribution] >> get_training_set
        get_training_set >> prep_training_df >> train_model >> store_model >> write_training_config_info
