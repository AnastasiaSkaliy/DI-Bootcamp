CREATE TABLE locations (
    location_id INTEGER PRIMARY KEY,
    address TEXT,
	district TEXT,
    city TEXT,
	country TEXT
);

CREATE TABLE branches(
	branch_id INTEGER PRIMARY KEY,
	location_id INTEGER,
	branch_name TEXT,
	location TEXT,
    contact_info TEXT,
	FOREIGN KEY(location_id) REFERENCES locations(location_id)
);

CREATE TABLE missions (
    mission_id INTEGER PRIMARY KEY,
	mission TEXT,
    mission_description TEXT,
	volunteering_time TEXT,
	car_needed BOOLEAN
);

CREATE TABLE branch_mission (
    branch_id INTEGER,
    mission_id INTEGER,
    FOREIGN KEY(branch_id) REFERENCES branches(branch_id),
    FOREIGN KEY(mission_id) REFERENCES missions(mission_id),
    PRIMARY KEY(branch_id, mission_id)
);
--add max numbers of voluntiers needed to missions column
ALTER TABLE missions DROP COLUMN max_volunteers;
ALTER TABLE missions DROP COLUMN mission_description;
ALTER TABLE missions ADD COLUMN mission TEXT;
ALTER TABLE missions ADD COLUMN mission_description TEXT;
ALTER TABLE missions ADD max_volunteers INTEGER;

--see all the data
SELECT * FROM missions
SELECT * FROM locations ORDER BY location_id
SELECT * FROM branches
SELECT * FROM branch_mission
DELETE FROM missions;

SELECT * FROM branches br, (SELECT * FROM locations loc WHERE location_id = loc.location_id) s WHERE br.location_id = s.location_id

-- ALTER TABLE locations ADD COLUMN temp_district TEXT;
-- ALTER TABLE locations DROP COLUMN temp_district;
-- ALTER TABLE locations DROP COLUMN country;
-- ALTER TABLE locations DROP COLUMN city;
-- ALTER TABLE locations DROP COLUMN district;
-- ALTER TABLE locations ADD COLUMN city TEXT;
-- ALTER TABLE locations ADD COLUMN district TEXT;
-- ALTER TABLE locations ADD COLUMN country TEXT;


--delete location column in branches table
ALTER TABLE branches DROP COLUMN location;
--update area 
UPDATE locations SET district = 'Tel Aviv District' WHERE location_id = 1


--create table missions one more time
ALTER TABLE branch_mission DROP CONSTRAINT branch_mission_mission_id_fkey;
DROP TABLE missions;
DELETE FROM missions;

--runner (only check purpuse)
SELECT m.mission_id, m.mission, m.mission_description
FROM missions m
JOIN branch_mission bm ON m.mission_id = bm.mission_id
JOIN branches b ON bm.branch_id = b.branch_id
WHERE b.location_id IN (1,2,3)
AND m.car_needed = TRUE
AND m.mission_id IN (4,5,6,1)
group by m.mission_id;
--check
SELECT * FROM missions WHERE volunteering_time LIKE '%Weekends%';
SELECT * FROM locations WHERE district LIKE %s;
--create table volunteers

CREATE TABLE volunteers (
    id serial PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    phone TEXT NOT NULL,
	mission_id INTEGER,
    mission TEXT NOT NULL
);
SELECT * FROM volunteers;

