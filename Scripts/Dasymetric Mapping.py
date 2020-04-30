# Flood Vulnerability Index Model

# This model utilizes tax parcel, land use, flood hazard, shelter location,
# and census data to create a flood vulnerability index for Houston, Texas.
# dasymetric mapping is utilized to disaggregate tract-level population 
# estimates to the parcel level, based on weight of parcel code.
# Populated parcels are then selected where they intersect flood hazard
# and shelter need areas. The results of the dasymetric analysis are ranked
# by percentile and combined with social justice data calculated through 
# principal component analysis in a different script. Factor weights were
# determined through analytical hierarchy process (AHP). The final output of this 
# model is a tract-level vulnerability index score, which is added to the 
# input census tract boundary feature class as a new field.

import arcpy as ap
from scipy import stats

ap.env.overwriteOutput = True

ap.env.workspace = r'C:\Users\jj09443\working\Houston\Final Thesis Data and Docs\Houston\Houston.gdb'

# The input spatial data for the model is listed below.

# Land use parcels containing 4-digit numeric land use codes are used to 
# identify large residential structures.

land_use = "COH_LAND_USE" 

# Tax parcel data containing 2-character alphanumeric tax codes are used
# to identify smaller residential parcels.

parcels = "Parcels"

# Final index scores are stored in the census tracts input feature class.

tracts = "Tracts"

# 100-year floodplain also known as special flood hazard area (SFHA) are
# used to demarcate areas of flood risk.

SFHA = "SFHA" 

# FEMA shelter points are used to identify local shelter locations.

shelters = "Shelters"

# The population table contains American Community Survey (ACS) 
# population estimates for each census tract.

pop_table = "TotalPop"

# The population field from the population table is added to the 
# tract feature class table.

pop_field = "TotalPop"

# The tract ID or GEOID is utilized for various join operations.

tract_id = "TRACT"

# Individual scores are calculated for flood risk and shelter
# accessibility, while the social justice score is provided
# from a seperate analysis. All three scores are weighted
# and combined for the final FV score.

FR_Score = "FR_Score"

SA_Score = "SA_Score"

SJ_Score = "SJ_Pctile"

FVI_Score = "FVI_SCore"

# Factor weights were determined through an AHP.

factor_weights = [52.7, 14.0, 33.3]

# A unique weight is assigned to each residential land use or state 
# classification code based on structure size and estimated residential
# population density.

lu_codes = [['4209', 12], ['4211', 24], ['4212', 48], ['4213', 24],
             ['4214', 96], ['4221', 48], ['4222', 48], ['4313', 48], 
             ['4316', 48], ['4319', 48], ['4670', 96], ['4613', 96]]

state_codes = [['A1', 1], ['A2', 1], ['B2', 2], 
                ['B3', 3], ['B4', 4], ['E1', 1]]

# Two functions in this model utilize a field calculator codeblock function.

codeblock = """

# This function returns 0 if an input value is null, and if the value is present, 
# multiplies that value by a specified weight.

def getcount(count, weight):
    if count is None:
        return 0
    else:
        return count * weight

# This function simply returns 0 for a null value and the value if it exists.

def NoNulls(v):
    
    if v is None:
        return 0
    else:
        return v

"""


# The JoinField function takes an input feature class and join table, and 
# creates a new field in the input table from a field in the join table, with
# nulls replaced by zeros. For this model, the join field for both inputs is
# always the tract ID/GEOID.

def JoinField(in_fc, join_table, in_field, out_field, join_field)
    """This function can be used to create a new field in a feature class
    table from a field in a joined table. Unlike the ArcPy JoinField_management
    function, this allows for a different name to be given to the output field
    This function takes 5 arguments:
    in_fc - input feature class
    join_table - table to be joined to feature class
    in_field - field in join_table to be added to in_fc
    out_field - name of new field in in_fc
    join_field - field on which the two tables will be joined. Must be the same 
    name for both datasets"""
    ap.AddField_management(in_fc, out_field, "DOUBLE")
    ap.MakeFeatureLayer_management(in_fc, "layer")
    ap.AddJoin_management("layer", join_field, join_table, join_field)
    ap.CalculateField_management(
        "layer", out_field, 
        "NoNulls(!{}.{}!)".format(join_table, in_field), 
        "PYTHON3", codeblock)
    ap.Delete_management("layer")

# JoinField is used to create the tract population field.

JoinField(tracts, pop_table, pop_field, pop_field, tract_id)

# Some multi-unit land use parcels overlap with numerous single-family 
# parcels. In this case, the state classifications are a more accurate 
# indicator of the number of people living in a parcel, and the coincident
# land use parcel must be removed.

ap.MakeFeatureLayer_management(
    land_use, "lu_lyr", 
    """"LANDUSE_CD" IN (
        '4209', '4211', '4212', 
        '4213', '4214', '4221', 
        '4222', '4313', '4316', 
        '4319', '4613', '4670')""")

ap.CopyFeatures_management("lu_lyr", "lu_res")
ap.Delete_management("lu_lyr")
ap.MakeFeatureLayer_management("lu_res", "res_lyr")
ap.MakeFeatureLayer_management(
    parcels, "parcel_lyr", 
    """"StClsCode" IN ('A1', 'A2', 
    'B2', 'B3', 'B4')""")

ap.SelectLayerByLocation_management(
    "res_lyr", "ARE_IDENTICAL_TO", "parcel_lyr")
ap.DeleteFeatures_management("res_lyr")
ap.Delete_management("res_lyr")
ap.Delete_management("parcel_lyr")

# The GetCodeCounts function uses summary statistics to calculate counts for
# each different parcel code in each tract. Tracts are spatially joined to 
# parcels in order to assign the tract ID field to each parcel Weighted counts 
# are then calculated by multiplying each count by its respective parcel code weight. 
# All weighted counts for each tract are then summed for the total weighted count.

def GetCodeCounts(
        parcel_fc, tract_fc, joined_parcels, 
        code_field, tract_id, sum_field, code_list):
    """This function generates tract-level weighted code counts from Houston
    parcel data. Weights are determined by estimated housing size. The function
    takes 7 arguments:

    parcel_fc - input parcel feature class
    tract_fc - census tract feature class
    joined_parcels - name of join feature class created from spatial join of 
    tracts and parcels
    code_field - name of input parcel code field
    tract_id - unique identifier field for tracts (GEOID)
    sum_field - name of field containing weighted code counts
    code_list - list containing parcel codes and thier associated weights,
    entered as [[code1, weight1], [code2, weight2], etc]

    The final output is a weighted code count field in the tract feature class
    """
    ap.SpatialJoin_analysis(
        parcel_fc, tract_fc, joined_parcels, 
        "JOIN_ONE_TO_ONE", "KEEP_ALL", "#", "HAVE_THEIR_CENTER_IN")

    for code in code_list:
        ap.MakeFeatureLayer_management(
            joined_parcels, "parcel_lyr", 
            code_field + """ = '{}'""".format(code[0], code_field))

        ap.Statistics_analysis(
            "parcel_lyr", "stats_{}".format(code[0]),
             [[code_field, "COUNT"]],  tract_id)

        ap.Delete_management("parcel_lyr")


    ap.MakeFeatureLayer_management(tract_fc, "tract_lyr")

    for code in code_list:
        ap.AddField_management(
            "tract_lyr", "Weighted_{}".format(code[0]), "LONG")
        ap.AddJoin_management(
            "tract_lyr", tract_id, 
            "stats_{}".format(code[0]), tract_id)
        ap.CalculateField_management(
            "tract_lyr", "Weighted_{}".format(code[0]), 
            "getcount(!stats_{}.COUNT_{}!, {})".format(code[0], 
                                                        code_field,
                                                        code[1]), 
            "PYTHON3", codeblock)
        ap.RemoveJoin_management("tract_lyr")

    ap.Delete_management("tract_lyr")   
    sum_codes= []
    for code in code_list:
        sum_codes.append('!Weighted_{}!'.format(code[0]))
    weighted_codes  = str(sum_codes).replace("'", "")
    
    ap.AddField_management(tract_fc, sum_field, "LONG")
    ap.CalculateField_management(
        tract_fc, sum_field, 
        "sum({})".format(weighted_codes), "PYTHON3")

# GetCodeCounts is applied to both land use and tax parcels, to get weighted counts 
# from both datasets. The two counts for each tract are then combined for the total 
# weighted residential parcel count.

GetCodeCounts(
    "lu_res", tracts, "lu_join", 'LANDUSE_CD', 
    tract_id, "LU_WeightedSum", lu_codes)

GetCodeCounts(
    parcels, tracts, "parcel_join","StClsCode", 
    tract_id, "Parcels_WeightedSum", state_codes)


ap.AddField_management(tracts, "WeightedSum_Total", "LONG")
ap.CalculateField_management(
    tracts, "WeightedSum_Total", 
    '!LU_WeightedSum! + !Parcels_WeightedSum!', "PYTHON3")

# The parcels containing the tract ID field created through GetCodeCounts are
# merged to create a single residential parcel layer. A 'Res_Units' field is
# added to the feature class table, which is then populated with the assigned
# weight for a given parcel's classification code.

ap.Merge_management(["lu_join", "parcel_join"], "All_Res")
ap.AddField_management("All_Res", "Res_Units", "LONG")

ap.MakeFeatureLayer_management("All_Res", "res_lyr")

for code in lu_codes:
    ap.SelectLayerByAttribute_management(
        "res_lyr", "NEW_SELECTION", 
        """"LANDUSE_CD" = '{}'""".format(code[0]))
    ap.CalculateField_management(
        "res_lyr", "Res_Units", code[1], "PYTHON3")
    
for code in state_codes:
    ap.SelectLayerByAttribute_management(
        "res_lyr", "NEW_SELECTION", 
        """"StClsCode" = '{}'""".format(code[0]))
    ap.CalculateField_management(
        "res_lyr", "Res_Units", code[1], "PYTHON3")

# Tract populations, weighted counts, and the Res_Units field created above 
# are used to estimate each parcel's population. Tract populations are divided 
# by weighted counts to get a 'people per unit' (PPU) field, which is then 
# multiplied by Res_Units to calculate the parcel population.

def ParcelPopulation(
        tract_fc, parcel_fc, pop_field,
         count_field, res_units, tract_id):
    
    """This function estimates the residential populations for parcel features,
    based on weighted parcel code counts and tract-level populations. It takes
    6 arguments:
    
    tract_fc - census tract feature class with population and weighted count fields
    parcel_fc - residential parcel feature class
    pop_field - population field in tract table
    count_field - weighted code count field in tract table
    res_units - field in parcel_fc containing estimated number of residential units 
    (code weight)
    tract_id - Unique identifier field for tracts (GEOID)
    
    The output is a parcel population field in the input parcel feature class"""

    ap.AddField_management(tract_fc, "PPU", "DOUBLE")
    ap.CalculateField_management(
        tract_fc, "PPU", 
        "!{}!/!{}!".format(pop_field, count_field))
    ap.AddField_management(parcel_fc, "ParcelPop", "DOUBLE")
    ap.MakeFeatureLayer_management(parcel_fc, "parcel_lyr")
    ap.AddJoin_management("parcel_lyr", tract_id, tract_fc, tract_id)
    ap.AddField_management("parcel_lyr", "PPU", "DOUBLE")
    ap.CalculateField_management(
        "parcel_lyr", "PPU", 
        "!{}.PPU!".format(tracts), "PYTHON3")
    ap.Delete_management("parcel_lyr")
    ap.CalculateField_management(
        parcel_fc, "ParcelPop", 
        "!{}!*!PPU!".format(res_units), "PYTHON3")

ParcelPopulation(
    tracts, "All_Res", "TotalPop", 
    "WeightedSum_Total", "Res_Units", tract_id)

# Areas of shelter need (shelter inaccessibility) are defined as areas
# where at-risk populations are not located within one mile of a shelter.
# These areas are created as all SFHA areas which do not intersect a
# shelter buffer.

ap.Buffer_analysis(shelters, "Shelter_Buffers", "1 MILE")
ap.Erase_analysis(SFHA, "Shelter_Buffers", "Shelter_Erase")

# Flood risk and shelter accessibility are computed at the tract level
# through a selection of residential parcel within flood risk and shelter
# need polygon layers

def DasymetricSelection(
        parcel_fc, pop_field, 
        tract_id, select_fc, out_table):
    """This function uses parcel-level populations to determine
    tract-level populations living within certain boundaries. 
    It takes 5 arguments:
    
    parcel_fc - input parcel feature class
    pop_field - parcel population field
    tract_id - tract ID (GEOID) field in the parcel table
    select_fc - feature class demarcating boundaries for selection
    out_table - name of the output table with tract-level dasymetric 
    population estimates"""
    ap.MakeFeatureLayer_management(parcel_fc, "res_lyr")
    ap.SelectLayerByLocation_management(
        "res_lyr", "INTERSECT", select_fc)
    ap.Statistics_analysis(
        "res_lyr", out_table, 
        [[pop_field, "SUM"]], tract_id)


DasymetricSelection(
    "All_Res", "ParcelPop", 
    tract_id, "Shelter_Erase", "NoShelter")

DasymetricSelection(
    "All_Res", "ParcelPop", 
    tract_id, "SFHA", "FloodHazard")

# JoinField is used to add the population fields from the statistics tables
# created through the DasymetricSelectionFunction to the census tract layer.

JoinField(
    tracts, "FloodHazard", 
    "SUM_ParcelPop", "AtRisk", tract_id)

JoinField(
    tracts, "NoShelter", 
    "SUM_ParcelPop", "ShelterInacc", tract_id)

# The GetPercentile uses the scipy module to calculate the percentile score
# for each record of an input field, counting nulls and zeros as 0. This
# function is used to standardize the dasymetric population counts,
# which can then be combined with the social justice field, which
# is also ranked by percentiles.

def GetPercentile(in_fc, in_field, out_field):
    """This function computes percentile scores for each value in an input 
    table field, and writes them to a new field in the same table. It takes
    3 arguments:
    
    in_fc - input feature class or table
    in_field - field to be ranked
    out_field - name of output percentile score field"""

    ap.MakeTableView_management(
        in_fc, "table_view", 
        '{0} IS NOT NULL AND {0} <>0'.format(in_field))

    ta = ap.da.TableToNumPyArray("table_view", [in_field])

    array = ta[in_field]

    ap.AddField_management(in_fc, out_field, "DOUBLE")

    cursor = ap.da.UpdateCursor(in_fc, [in_field, out_field])

    for row in cursor:
        if row[0] !=0 and not row[0] is None:
            row[1] = stats.percentileofscore(array, row[0])
        else:
            row[1] = 0
        cursor.updateRow(row)
    
GetPercentile(tracts, "AtRisk", "FR_Score")
GetPercentile(tracts, "ShelterInacc", "SA_SCore")
#GetPercentile(tracts, SJ_Score, "SJ_Score")

score_fields = [FR_Score, SA_Score, SJ_Score]

# The finalScore function takes two input lists of equal length for the 
# three factor scores and their respective weights. Each factor is weighted
# and then all three are combined for the final FVI score.

def FinalScore(in_fc, in_fields, weights, fv_score):
    """This function calculates a weighted average based on a list of input
    fields and a corresponding list of weights. It takes 4 arguments:
    
    in_fc - input feature class or table containing fields to be averaged
    in_fields - list of input fields
    weights - list of weights for each input field
    fv_score - name of final score field containing weighted averages"""

    weightcalcs = [weight/100 for weight in weights]
    fieldcalcs = ["!" + field + "!" for field in in_fields]
    ap.AddField_management(in_fc, fv_score, "DOUBLE")
    ap.CalculateField_management(
        in_fc, fv_score, 
        "(" + fieldcalcs[0] + "*{})".format(weightcalcs[0]) + 
        " + (" + fieldcalcs[1] + "*{})".format(weightcalcs[0]) + 
        " + (" + fieldcalcs[2] + "*{})".format(weightcalcs[0]))


FinalScore(tracts, score_fields, factor_weights, FVI_Score)

