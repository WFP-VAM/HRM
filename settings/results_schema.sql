CREATE TABLE "public".results (
	id serial NOT NULL,
	run_date date NULL,
	config_id int4 NULL,
	r2pearson float8 NULL,
	r2pearson_var float8 NULL,
	mape float8 NULL,
	mape_var varchar NULL,
	r2 float4 NULL,
	r2_var float4 NULL,
	description varchar NULL,
	PRIMARY KEY (id),
	FOREIGN KEY (config_id) REFERENCES config(id)
);
