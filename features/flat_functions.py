from feature import FeatureBase
import inflect
import column
import pdb
import ntiDSMGlobal as ntiGlb

p = inflect.engine()

class FlatFeature(FeatureBase):
    # def col_allowed(self, col):


    def apply(self,fk, inverse=False):
        """
        brings in all flat features according to the foreign key provided

        inverse option flatten in opposite direction of fk. this is used when the foreign key represents a one to one relationship
        """
        table = self.db.tables[fk.parent.table.name]
        parent_table = self.db.tables[fk.column.table.name]

        ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function : table = %s & parent_table = %s, inverse = %s" %(table.name,parent_table.name,str(inverse)))

        if inverse:
            ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function: since inverse = %s, interchanging parent and table name" %(str(inverse)))
            table, parent_table = parent_table, table
            ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function : NOW table = %s & parent_table = %s" %(table.name,parent_table.name))

        child_column = fk.parent
        parent_column = fk.column
        ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function : table = %s & parent_table = %s, inverse = %s -> child_column_name = %s & parent_column_name = %s" %(table.name,parent_table.name,str(inverse),child_column.name,parent_column.name))

        if inverse:
            ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function: since inverse = %s, interchanging child_column and parent_column" %(str(inverse)))
            child_column,parent_column = parent_column, child_column
            ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function : NOW table = %s & parent_table = %s, inverse = %s -> child_column_name = %s & parent_column_name = %s" %(table.name,parent_table.name,str(inverse),child_column.name,parent_column.name))

        #if the fk has a special column name in parent table, keep it. otherwise use the name of the foreign table\
        if child_column.name != parent_column.name:
            prefix = parent_column.name
        else:
            prefix = parent_column.table.name
            singular = p.singular_noun(prefix)
            if singular:
                prefix = singular



        prefix += "."
        ntiGlb.ntiDSMGlobalObj.log("In Flat Feature Apply function : After deliberation, final prefix = %s" %(prefix))

        to_add = parent_table.get_column_info(ignore_relationships=True)
        ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function : to add column info obtained = %s" %(str(to_add)))
        # pdb.set_trace()
        for col in to_add:
            new_metadata = col.copy_metadata()

            path_add = {
                'base_column': col,
                'feature_type' : 'flat',
                'feature_type_func' : None
            }


            real_name = prefix + col.metadata['real_name']
            new_metadata.update({ 
                'path' : new_metadata['path'] + [path_add],
                'real_name' : real_name
            })

            #don't make feature if table has it
            if table.has_feature(real_name): 
                ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function : feature = %s already exists in table = %s, skipping it...." %(str(real_name),table.name))
                continue

            path = col.metadata["path"]
            if  path!= [] and path[-1]["base_column"].dsm_table == table and path[-1]["feature_type"] == "flat":
                print "flat feature from target table"
                print col, table
                print
                return False

            new_col = column.DSMColumn(col.column, table, new_metadata)
            ntiGlb.ntiDSMGlobalObj.log("In Flat Features Apply function : table = %s -> new column made , column name = %s, will be added now" %(table.name,new_col.name))

            # pdb.set_trace()
            table.add_column(new_col)
