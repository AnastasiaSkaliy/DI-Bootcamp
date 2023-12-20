CREATE TABLE menu_items (
	item_id serial PRIMARY KEY,
	item_name VARCHAR (30) NOT NULL,
	item_price SMALLINT DEFAULT 0
);

SELECT * FROM menu_items