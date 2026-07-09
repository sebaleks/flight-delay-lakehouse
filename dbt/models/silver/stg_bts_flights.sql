{{
    config(
        materialized='table',
        partition_by={'field': 'flight_date', 'data_type': 'date', 'granularity': 'month'},
        cluster_by=['origin', 'dest', 'reporting_airline'],
    )
}}

-- One row per flight leg, cleaned and typed from the raw bronze external
-- table. Deliberately a full-column SUPERSET including post-departure and
-- arrival outcomes (dep_delay, taxi/wheels times, actual elapsed/air time,
-- arrival fields): analytics uses delay outcomes, and the leakage boundary
-- is enforced at the gold ML feature mart (CLAUDE.md §9), not in silver.
--
-- Typing: hhmm local times -> TIME ('2400' means midnight -> 00:00);
-- 0/1 flags -> BOOL (NULL preserved, e.g. arr_del15 on cancelled flights);
-- delays/durations -> FLOAT64 minutes; ids/groups -> INT64.
-- Bronze quirks dropped: Year_file/Month_file (duplicates of the hive
-- partition keys year/month, which are kept) and trailing_empty.

with source as (

    select * from {{ source('bronze', 'bts_on_time_performance') }}

),

typed as (

    select
        year,
        month,
        safe_cast(safe_cast(Quarter as float64) as int64) as quarter,
        safe_cast(safe_cast(DayofMonth as float64) as int64) as day_of_month,
        safe_cast(safe_cast(DayOfWeek as float64) as int64) as day_of_week,
        safe_cast(FlightDate as date) as flight_date,
        upper(nullif(trim(Reporting_Airline), '')) as reporting_airline,
        safe_cast(safe_cast(DOT_ID_Reporting_Airline as float64) as int64) as dot_id_reporting_airline,
        nullif(trim(IATA_CODE_Reporting_Airline), '') as iata_code_reporting_airline,
        nullif(trim(Tail_Number), '') as tail_number,
        safe_cast(safe_cast(Flight_Number_Reporting_Airline as float64) as int64) as flight_number_reporting_airline,
        safe_cast(safe_cast(OriginAirportID as float64) as int64) as origin_airport_id,
        safe_cast(safe_cast(OriginAirportSeqID as float64) as int64) as origin_airport_seq_id,
        safe_cast(safe_cast(OriginCityMarketID as float64) as int64) as origin_city_market_id,
        upper(nullif(trim(Origin), '')) as origin,
        nullif(trim(OriginCityName), '') as origin_city_name,
        nullif(trim(OriginState), '') as origin_state,
        nullif(trim(OriginStateFips), '') as origin_state_fips,
        nullif(trim(OriginStateName), '') as origin_state_name,
        safe_cast(safe_cast(OriginWac as float64) as int64) as origin_wac,
        safe_cast(safe_cast(DestAirportID as float64) as int64) as dest_airport_id,
        safe_cast(safe_cast(DestAirportSeqID as float64) as int64) as dest_airport_seq_id,
        safe_cast(safe_cast(DestCityMarketID as float64) as int64) as dest_city_market_id,
        upper(nullif(trim(Dest), '')) as dest,
        nullif(trim(DestCityName), '') as dest_city_name,
        nullif(trim(DestState), '') as dest_state,
        nullif(trim(DestStateFips), '') as dest_state_fips,
        nullif(trim(DestStateName), '') as dest_state_name,
        safe_cast(safe_cast(DestWac as float64) as int64) as dest_wac,
        safe.parse_time('%H%M', if(nullif(trim(CRSDepTime), '') = '2400', '0000', lpad(nullif(trim(CRSDepTime), ''), 4, '0'))) as crs_dep_time,
        safe.parse_time('%H%M', if(nullif(trim(DepTime), '') = '2400', '0000', lpad(nullif(trim(DepTime), ''), 4, '0'))) as dep_time,
        safe_cast(DepDelay as float64) as dep_delay,
        safe_cast(DepDelayMinutes as float64) as dep_delay_minutes,
        safe_cast(DepDel15 as float64) = 1 as dep_del15,
        safe_cast(safe_cast(DepartureDelayGroups as float64) as int64) as departure_delay_groups,
        nullif(trim(DepTimeBlk), '') as dep_time_blk,
        safe_cast(TaxiOut as float64) as taxi_out,
        safe.parse_time('%H%M', if(nullif(trim(WheelsOff), '') = '2400', '0000', lpad(nullif(trim(WheelsOff), ''), 4, '0'))) as wheels_off,
        safe.parse_time('%H%M', if(nullif(trim(WheelsOn), '') = '2400', '0000', lpad(nullif(trim(WheelsOn), ''), 4, '0'))) as wheels_on,
        safe_cast(TaxiIn as float64) as taxi_in,
        safe.parse_time('%H%M', if(nullif(trim(CRSArrTime), '') = '2400', '0000', lpad(nullif(trim(CRSArrTime), ''), 4, '0'))) as crs_arr_time,
        safe.parse_time('%H%M', if(nullif(trim(ArrTime), '') = '2400', '0000', lpad(nullif(trim(ArrTime), ''), 4, '0'))) as arr_time,
        safe_cast(ArrDelay as float64) as arr_delay,
        safe_cast(ArrDelayMinutes as float64) as arr_delay_minutes,
        safe_cast(ArrDel15 as float64) = 1 as arr_del15,
        safe_cast(safe_cast(ArrivalDelayGroups as float64) as int64) as arrival_delay_groups,
        nullif(trim(ArrTimeBlk), '') as arr_time_blk,
        safe_cast(Cancelled as float64) = 1 as cancelled,
        nullif(trim(CancellationCode), '') as cancellation_code,
        safe_cast(Diverted as float64) = 1 as diverted,
        safe_cast(CRSElapsedTime as float64) as crs_elapsed_time,
        safe_cast(ActualElapsedTime as float64) as actual_elapsed_time,
        safe_cast(AirTime as float64) as air_time,
        safe_cast(safe_cast(Flights as float64) as int64) as flights,
        safe_cast(Distance as float64) as distance,
        safe_cast(safe_cast(DistanceGroup as float64) as int64) as distance_group,
        safe_cast(CarrierDelay as float64) as carrier_delay,
        safe_cast(WeatherDelay as float64) as weather_delay,
        safe_cast(NASDelay as float64) as nas_delay,
        safe_cast(SecurityDelay as float64) as security_delay,
        safe_cast(LateAircraftDelay as float64) as late_aircraft_delay,
        safe.parse_time('%H%M', if(nullif(trim(FirstDepTime), '') = '2400', '0000', lpad(nullif(trim(FirstDepTime), ''), 4, '0'))) as first_dep_time,
        safe_cast(TotalAddGTime as float64) as total_add_g_time,
        safe_cast(LongestAddGTime as float64) as longest_add_g_time,
        safe_cast(safe_cast(DivAirportLandings as float64) as int64) as div_airport_landings,
        safe_cast(DivReachedDest as float64) = 1 as div_reached_dest,
        safe_cast(DivActualElapsedTime as float64) as div_actual_elapsed_time,
        safe_cast(DivArrDelay as float64) as div_arr_delay,
        safe_cast(DivDistance as float64) as div_distance,
        nullif(trim(Div1Airport), '') as div1_airport,
        safe_cast(safe_cast(Div1AirportID as float64) as int64) as div1_airport_id,
        safe_cast(safe_cast(Div1AirportSeqID as float64) as int64) as div1_airport_seq_id,
        safe.parse_time('%H%M', if(nullif(trim(Div1WheelsOn), '') = '2400', '0000', lpad(nullif(trim(Div1WheelsOn), ''), 4, '0'))) as div1_wheels_on,
        safe_cast(Div1TotalGTime as float64) as div1_total_g_time,
        safe_cast(Div1LongestGTime as float64) as div1_longest_g_time,
        safe.parse_time('%H%M', if(nullif(trim(Div1WheelsOff), '') = '2400', '0000', lpad(nullif(trim(Div1WheelsOff), ''), 4, '0'))) as div1_wheels_off,
        nullif(trim(Div1TailNum), '') as div1_tail_num,
        nullif(trim(Div2Airport), '') as div2_airport,
        safe_cast(safe_cast(Div2AirportID as float64) as int64) as div2_airport_id,
        safe_cast(safe_cast(Div2AirportSeqID as float64) as int64) as div2_airport_seq_id,
        safe.parse_time('%H%M', if(nullif(trim(Div2WheelsOn), '') = '2400', '0000', lpad(nullif(trim(Div2WheelsOn), ''), 4, '0'))) as div2_wheels_on,
        safe_cast(Div2TotalGTime as float64) as div2_total_g_time,
        safe_cast(Div2LongestGTime as float64) as div2_longest_g_time,
        safe.parse_time('%H%M', if(nullif(trim(Div2WheelsOff), '') = '2400', '0000', lpad(nullif(trim(Div2WheelsOff), ''), 4, '0'))) as div2_wheels_off,
        nullif(trim(Div2TailNum), '') as div2_tail_num,
        nullif(trim(Div3Airport), '') as div3_airport,
        safe_cast(safe_cast(Div3AirportID as float64) as int64) as div3_airport_id,
        safe_cast(safe_cast(Div3AirportSeqID as float64) as int64) as div3_airport_seq_id,
        safe.parse_time('%H%M', if(nullif(trim(Div3WheelsOn), '') = '2400', '0000', lpad(nullif(trim(Div3WheelsOn), ''), 4, '0'))) as div3_wheels_on,
        safe_cast(Div3TotalGTime as float64) as div3_total_g_time,
        safe_cast(Div3LongestGTime as float64) as div3_longest_g_time,
        safe.parse_time('%H%M', if(nullif(trim(Div3WheelsOff), '') = '2400', '0000', lpad(nullif(trim(Div3WheelsOff), ''), 4, '0'))) as div3_wheels_off,
        nullif(trim(Div3TailNum), '') as div3_tail_num,
        nullif(trim(Div4Airport), '') as div4_airport,
        safe_cast(safe_cast(Div4AirportID as float64) as int64) as div4_airport_id,
        safe_cast(safe_cast(Div4AirportSeqID as float64) as int64) as div4_airport_seq_id,
        safe.parse_time('%H%M', if(nullif(trim(Div4WheelsOn), '') = '2400', '0000', lpad(nullif(trim(Div4WheelsOn), ''), 4, '0'))) as div4_wheels_on,
        safe_cast(Div4TotalGTime as float64) as div4_total_g_time,
        safe_cast(Div4LongestGTime as float64) as div4_longest_g_time,
        safe.parse_time('%H%M', if(nullif(trim(Div4WheelsOff), '') = '2400', '0000', lpad(nullif(trim(Div4WheelsOff), ''), 4, '0'))) as div4_wheels_off,
        nullif(trim(Div4TailNum), '') as div4_tail_num,
        nullif(trim(Div5Airport), '') as div5_airport,
        safe_cast(safe_cast(Div5AirportID as float64) as int64) as div5_airport_id,
        safe_cast(safe_cast(Div5AirportSeqID as float64) as int64) as div5_airport_seq_id,
        safe.parse_time('%H%M', if(nullif(trim(Div5WheelsOn), '') = '2400', '0000', lpad(nullif(trim(Div5WheelsOn), ''), 4, '0'))) as div5_wheels_on,
        safe_cast(Div5TotalGTime as float64) as div5_total_g_time,
        safe_cast(Div5LongestGTime as float64) as div5_longest_g_time,
        safe.parse_time('%H%M', if(nullif(trim(Div5WheelsOff), '') = '2400', '0000', lpad(nullif(trim(Div5WheelsOff), ''), 4, '0'))) as div5_wheels_off,
        nullif(trim(Div5TailNum), '') as div5_tail_num
    from source

),

-- bronze is landed per month and currently has zero duplicate keys; this is
-- a safety net so a re-landed or overlapping partition can never fan out
-- downstream joins. Preference: rows WITH arrival data first, then a
-- content-hash tiebreak for a deterministic total order (identical rows are
-- interchangeable). Bronze carries no load-recency signal, so a corrected
-- re-land must REPLACE its partition (ingestion.bts --force), not append.
deduped as (

    select *
    from typed
    qualify row_number() over (
        partition by
            flight_date, reporting_airline, flight_number_reporting_airline,
            origin, dest, crs_dep_time
        order by
            arr_time is null, arr_time, tail_number,
            farm_fingerprint(to_json_string(typed))
    ) = 1

)

select * from deduped
