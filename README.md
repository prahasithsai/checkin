# checkin
* Project Title: “Customer Check In Prediction”
* Project Mangement Methodology used: CRISP ML (Q)
* Scope of the Project: In this Business Problem, based on the past hotel booking information details of the customer i.e., dataset given by the client, using the data  analysis & visualization, data wrangling & build the model that predicts the customer who is going to be check in to the hotel room.
*********************************************************************************************************************************************************************** 
* Step (1) - Business Understanding & Data Understanding:
* Business Objective: To predict whether customer will check in to the Hotel room or not.
* Business Constriant: To choose most significant features.
* Data Collection & Data Types:   
*            **Column**           **Count**     **Non-Null**       **Dtype**  
          0   Unnamed: 0            82580         non-null           int64  
          1   ID                    82580         non-null           int64  
          2   Nationality           82580         non-null           object 
          3   Age                   78834         non-null           float64
          4   DaysSinceCreation     82580         non-null           int64  
          5   AverageLeadTime       82580         non-null           int64  
          6   LodgingRevenue        82580         non-null           float64
          7   OtherRevenue          82580         non-null           float64
          8   BookingsCanceled      82580         non-null           int64  
          9   BookingsNoShowed      82580         non-null           int64  
          10  BookingsCheckedIn     82580         non-null           int64  
          11  PersonsNights         82580         non-null           int64  
          12  RoomNights            82580         non-null           int64  
          13  DaysSinceLastStay     82580         non-null           int64  
          14  DaysSinceFirstStay    82580         non-null           int64  
          15  DistributionChannel   82580         non-null           object 
          16  MarketSegment         82580         non-null           object 
          17  SRHighFloor           82580         non-null           int64  
          18  SRLowFloor            82580         non-null           int64  
          19  SRAccessibleRoom      82580         non-null           int64  
          20  SRMediumFloor         82580         non-null           int64  
          21  SRBathtub             82580         non-null           int64  
          22  SRShower              82580         non-null           int64  
          23  SRCrib                82580         non-null           int64  
          24  SRKingSizeBed         82580         non-null           int64  
          25  SRTwinBed             82580         non-null           int64  
          26  SRNearElevator        82580         non-null           int64  
          27  SRAwayFromElevator    82580         non-null           int64  
          28  SRNoAlcoholInMiniBar  82580         non-null           int64  
          29  SRQuietRoom           82580         non-null           int64  
           *  RangeIndex: 82580 entries, 0 to 82579, Data columns (total 30 columns)
           *  dtypes: float64(3), int64(24), object(3)
***********************************************************************************************************************************************************************
* Step (2) - Data Preprocessing/EDA/Feature Engineering:
* Data Preprocessing:
* 1) Dummy variable creation: Using Label Encoding
* 2) Checking for zero variance features: With thresholdlimit=0
* 3) Handling missing values: Used median imputation 
