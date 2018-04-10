CREATE TABLE "public".config_new (
	dataset_filename varchar NULL,
	satellite_grid varchar NULL,
	satellite_step int2 NULL DEFAULT 0,
	sentinel_config json null,
	id serial NOT NULL,
	model_grid_parameters json NULL,
	"indicator" varchar NULL,
	"comment" varchar NULL,
	"output" varchar NOT NULL DEFAULT 'regression'::character varying,
	land_use_raster varchar NULL,
	PRIMARY KEY (id)
);