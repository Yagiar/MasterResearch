--liquibase formatted sql

--changeset uavdet:001-schema
--comment: схема uavdet для результатов пайплайна обнаружения БПЛА
CREATE SCHEMA IF NOT EXISTS uavdet;
--rollback DROP SCHEMA IF EXISTS uavdet CASCADE;
