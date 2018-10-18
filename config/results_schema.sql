CREATE TABLE "public".results (
	id serial NOT NULL,
	run_date date NULL,
	config_id int4 NULL,
	description varchar NULL,
	r2 float4 NULL,
	r2_sd float4 NULL,
	r2_knn float4 NULL,
	r2_sd_knn float4 NULL,
	r2_features float4 NULL,
	r2_sd_features float4 NULL,
	mape_rmsense float4 NULL,
	PRIMARY KEY (id),
	FOREIGN KEY (config_id) REFERENCES config_new(id)
)
