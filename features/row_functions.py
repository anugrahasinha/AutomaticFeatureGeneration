import sqlalchemy.dialects.mysql.base as column_datatypes
from feature import FeatureBase

import ntiDSMGlobal as ntiGlb

class MysqlRowFunc(FeatureBase):
    name = "MysqlRowFunc"
    func = None
    numeric = False
    categorical = False
    col_type = column_datatypes.INTEGER.__visit_name__

    def get_allowed_cols(self):
        pass

    def do_qry(self, target_table, set_vals):
        SET = []
        for s in set_vals:
            SET.append("`t`.%s = %s(`t`.%s)" % (s[0], self.func, s[1]))
        SET = ",".join(SET)
        ntiGlb.ntiDSMGlobalObj.log("Function = %s ---- ROW ANUGRAHA SINHA --- target_table = %s set_vals = %s" %(self.name,target_table,str(set_vals)))
        qry = """
            UPDATE `{target_table}` t
            SET {SET}
            """.format(target_table=target_table, SET=SET)
        #print("******** ANUGRAHA SINHA **********")
        #print(qry)
        ntiGlb.ntiDSMGlobalObj.log("Function = %s Final update query = %s" %(str(self.name)," ".join(qry.strip("\n").strip().split("\n"))),loglevel="DEBUG")
        
        self.db.execute(qry)
        ## Adding the queries to globalFeatureGenerationStatistics ##
        for s in set_vals:
            ntiGlb.ntiDSMGlobalObj.globalFeatureGenerationStatistics[target_table][s[0]]["SQL_query"] = " ".join(qry.strip("\n").strip().split("\n"))
        return qry

    def apply(self, table):
        to_add = []

        #create columns
        ntiGlb.ntiDSMGlobalObj.log("For function = %s, allowable column list = %s" %(self.name,str(self.get_allowed_cols(table))))
        for col in self.get_allowed_cols(table):
            ntiGlb.ntiDSMGlobalObj.log("For function = %s table = %s, working for column = %s" %(self.name,table.name,col.name))
            real_name = "%s(%s)" % (self.func,col.metadata["real_name"])
            
            new_metadata = col.copy_metadata()

            path_add = {
                'base_column': col,
                'feature_type' : 'row',
                'feature_type_func' : self.name
            }

            new_metadata.update({
                'real_name' : real_name,
                'numeric' : self.numeric,
                'categorical': self.categorical,
                'path' : new_metadata['path'] + [path_add],
            })

            #don't make feature if table has it
            if table.has_feature(real_name):
                ntiGlb.ntiDSMGlobalObj.log("For function = %s table = %s, already has this feature built, skipping it" %(self.name,table.name))
                continue

            new_table_name, new_col_name = table.create_column(self.col_type, metadata=new_metadata)
            ntiGlb.ntiDSMGlobalObj.log("For function = %s, new_table_name = %s and new_col_name = %s" %(self.name,new_table_name,new_col_name))
            to_add.append((col, (new_table_name, new_col_name)))

        ntiGlb.ntiDSMGlobalObj.log("For function = %s, final add column list = %s" %(self.name,str(to_add)))

        last_table_name = None
        set_vals = [] 
        for (col, (new_table_name, new_col_name)) in to_add:
            if last_table_name == None:
                last_table_name = new_table_name

            if last_table_name!=new_table_name:
                query_executed = self.do_qry(last_table_name, set_vals)
                last_table_name = new_table_name
                set_vals = []

            set_vals.append([new_col_name, col.column.name])

        if set_vals != []:
            self.do_qry(last_table_name, set_vals)
    

class TextLength(MysqlRowFunc):
    name = "text_length"
    func = "length"
    numeric = True
    categorical = False
    col_type = column_datatypes.INTEGER.__visit_name__

    def get_allowed_cols(self, table):
        return table.get_columns_of_type([column_datatypes.TEXT], ignore_relationships=True)

class Weekday(MysqlRowFunc):
    name = "weekday"
    func = "weekday"
    numeric = False
    categorical = True
    col_type = column_datatypes.INTEGER.__visit_name__

    def get_allowed_cols(self, table):
        return table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True)

class Month(MysqlRowFunc):
    name = "month"
    func = "month"
    numeric = False
    categorical = True
    col_type = column_datatypes.INTEGER.__visit_name__

    def get_allowed_cols(self, table):
        return table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True)
    


def apply_funcs(table):
    funcs = [TextLength, Weekday, Month]
    excluded = table.config.get("excluded_row_functions", [])
    included = table.config.get("included_row_functions", funcs) #if none included, include all
    ntiGlb.ntiDSMGlobalObj.log("As per config, excluded_row_functions = %s" %(str(excluded)))
    ntiGlb.ntiDSMGlobalObj.log("As per config, included_row_functions = %s" %(str(included)))
    included = set(included).difference(excluded)
    ntiGlb.ntiDSMGlobalObj.log("Finally included function (i.e. included - excluded) = %s" %(str(included)))
    for func in funcs:
        ## NTI Comment : There seems to be a problem here, trying to correct it ##
        ##             : included are, function names, which are classes,
        ##             : Correcting this to be linked with included.name
        included_names = [x.name for x in included]
        #if func.name in included:
        if func.name in included_names:
            ntiGlb.ntiDSMGlobalObj.log("Executing function name = %s, where in function = %s" %(func.name,func.func))
            ntiGlb.ntiDSMGlobalObj.log("Other variables in this function - numeric = %s, categorical = %s, column type = INTEGER" %(str(func.numeric),str(func.categorical)))
            func(table.db).apply(table)


# def convert_datetime_weekday(table):
#     for col in table.get_columns_of_type([column_datatypes.DATETIME, column_datatypes.DATE], ignore_relationships=True):
#         real_name = "DAY({col_name})".format(col_name=col.metadata['real_name'])
#         new_metadata = col.copy_metadata()
        
#         path_add = {
#                     'base_column': col,
#                     'feature_type' : 'row',
#                     'feature_type_func' : "weekday"
#                 }

#         new_metadata.update({ 
#             'path' : new_metadata['path'] + [path_add],
#             'numeric' : False,
#             'categorical' : True,
#             "real_name" : real_name
#         })


#         new_table_name, new_col_name = table.create_column(column_datatypes.INTEGER.__visit_name__, metadata=new_metadata)

#         params = {
#             target_table: new_table_name,
#             new_col_name : new_col_name,
#             src_table: col.column.table.name,
#             col_name : col.name
#         }

#         qry = """
#             UPDATE `{target_table}` t
#             set `{new_col_name}` = WEEKDAY({src_table}.`{col_name}`)
#             """ % (**params)
#         print qry
#         table.engine.execute(qry)

#TODO UPDATE EVERYTHING BELOW HERE



        
def add_ntiles(table, n=10):
    for col in table.get_numeric_columns(ignore_relationships=True):
        new_col = "[{col_name}]_decile".format(col_name=col.metadata['real_name'])
        new_metadata = col.copy_metadata()
        path_add = {
                    'base_column': col,
                    'feature_type' : 'row',
                    'feature_type_func' : "ntile"
                }

        new_metadata.update({ 
            'path' : new_metadata['path'] + [path_add],
            'numeric' : False,
            "real_name" : new_col
            # 'excluded_agg_funcs' : set(['sum']),
            # 'excluded_row_funcs' : set(['add_ntiles']),
        })

        if len(new_metadata['path']) > MAX_FUNC_TO_APPLY:
            continue

        new_col_name = table.create_column(column_datatypes.INTEGER.__visit_name__, metadata=new_metadata, flush=True)
        select_pk = ", ".join(["`%s`"%pk for pk in table.primary_key_names])

        where_pk = ""
        first = True
        for pk in table.primary_key_names:
            if not first:
                where_pk += " AND "
            where_pk += "`%s` = `%s`.`%s`" % (pk, table.name, pk)
            first = False

        qry = """
        UPDATE `{table}`
        SET `{table}`.`{new_col}` = 
        (
            select round({n}*(cnt-rank+1)/cnt,0) as decile from
            (
                SELECT  {select_pk}, @curRank := @curRank + 1 AS rank
                FROM   `{table}` p,
                (
                    SELECT @curRank := 0) r
                    ORDER BY `{col_name}` desc
                ) as dt,
                (
                    select count(*) as cnt
                    from `{table}`
                ) as ct
            WHERE {where_pk}
        );
        """.format(table=table.name, new_col=new_col_name, n=n, col_name=col.name, select_pk=select_pk, where_pk=where_pk)
        table.engine.execute(qry) #very bad, fix how parameters are substituted in
